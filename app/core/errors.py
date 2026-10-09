"""Format d'erreur unique de l'API : `{detail, error_code, request_id}` (+ `debug` en développement).

- `detail` : message en français destiné à l'utilisateur ;
- `error_code` : code stable, exploitable par le client (voir `ErrorCode`) ;
- `request_id` : identifiant de corrélation (`X-Request-ID`), à retrouver dans les logs ;
- `debug` : type/message de l'exception ou détail de validation, **uniquement si `APP_ENV=development`**.
"""

import logging
from enum import StrEnum
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.request_id import REQUEST_ID_HEADER, request_id_var

logger = logging.getLogger(__name__)

INTERNAL_ERROR_DETAIL = "Erreur interne inattendue"
_VALIDATION_LOC_PREFIXES = frozenset({"body", "query", "path", "header", "cookie"})


class ErrorCode(StrEnum):
    BAD_REQUEST = "bad_request"
    UNAUTHORIZED = "unauthorized"
    FORBIDDEN = "forbidden"
    NOT_FOUND = "not_found"
    METHOD_NOT_ALLOWED = "method_not_allowed"
    CONFLICT = "conflict"
    GONE = "gone"
    PAYLOAD_TOO_LARGE = "payload_too_large"
    RANGE_NOT_SATISFIABLE = "range_not_satisfiable"
    VALIDATION_ERROR = "validation_error"
    RATE_LIMITED = "rate_limited"
    QUOTA_EXCEEDED = "quota_exceeded"
    GEMINI_QUOTA = "gemini_quota"
    GEMINI_UNAVAILABLE = "gemini_unavailable"
    GEMINI_INVALID_RESPONSE = "gemini_invalid_response"
    OLLAMA_UNAVAILABLE = "ollama_unavailable"
    UPSTREAM_ERROR = "upstream_error"
    SERVICE_UNAVAILABLE = "service_unavailable"
    INTERNAL_ERROR = "internal_error"
    HTTP_ERROR = "http_error"


_DEFAULT_CODES: dict[int, ErrorCode] = {
    400: ErrorCode.BAD_REQUEST,
    401: ErrorCode.UNAUTHORIZED,
    403: ErrorCode.FORBIDDEN,
    404: ErrorCode.NOT_FOUND,
    405: ErrorCode.METHOD_NOT_ALLOWED,
    409: ErrorCode.CONFLICT,
    410: ErrorCode.GONE,
    413: ErrorCode.PAYLOAD_TOO_LARGE,
    416: ErrorCode.RANGE_NOT_SATISFIABLE,
    422: ErrorCode.VALIDATION_ERROR,
    429: ErrorCode.RATE_LIMITED,
    500: ErrorCode.INTERNAL_ERROR,
    502: ErrorCode.UPSTREAM_ERROR,
    503: ErrorCode.SERVICE_UNAVAILABLE,
}

# Libellés de repli quand une HTTPException n'a pas de `detail` exploitable (ex. 404/405 de routage).
_DEFAULT_DETAILS: dict[int, str] = {
    404: "Ressource introuvable",
    405: "Méthode non autorisée",
}


def _user_detail(status_code: int, detail: Any) -> str | None:
    """`detail` textuel fourni par l'application, ou None si absent / phrase HTTP par défaut
    de Starlette (« Not Found », « Method Not Allowed »…), qui n'est pas un message utilisateur."""
    if not isinstance(detail, str) or not detail:
        return None
    try:
        if detail == HTTPStatus(status_code).phrase:
            return None
    except ValueError:
        pass
    return detail


def default_error_code(status_code: int) -> ErrorCode:
    return _DEFAULT_CODES.get(status_code, ErrorCode.HTTP_ERROR)


class ApiError(HTTPException):
    """`HTTPException` portant un `error_code` explicite et un `debug` (jamais exposé en production)."""

    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: ErrorCode,
        *,
        headers: dict[str, str] | None = None,
        debug: Any = None,
    ) -> None:
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.error_code = error_code
        self.debug = debug


def exception_debug(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)}


def current_request_id(request: Request | None = None) -> str:
    if request is not None:
        request_id = getattr(request.state, "request_id", None)
        if request_id:
            return request_id
    return request_id_var.get()


def error_response(
    request: Request | None,
    status_code: int,
    detail: str,
    error_code: ErrorCode | str,
    *,
    headers: dict[str, str] | None = None,
    debug: Any = None,
    include_debug: bool = False,
) -> JSONResponse:
    request_id = current_request_id(request)
    content: dict[str, Any] = {"detail": detail, "error_code": str(error_code), "request_id": request_id}
    if include_debug and debug is not None:
        content["debug"] = jsonable_encoder(debug)
    return JSONResponse(
        status_code=status_code,
        content=content,
        headers={**(headers or {}), REQUEST_ID_HEADER: request_id},
    )


def _field_name(loc: tuple | list) -> str:
    parts = [str(p) for p in loc if not (isinstance(p, str) and p in _VALIDATION_LOC_PREFIXES)]
    return ".".join(parts) or "requête"


def validation_detail(errors: list[dict[str, Any]]) -> str:
    """Message lisible : les champs en cause, sans le jargon pydantic (réservé à `debug`)."""
    fields = list(dict.fromkeys(_field_name(err.get("loc", ())) for err in errors))
    if not fields:
        return "Requête invalide"
    if len(fields) == 1:
        return f"Requête invalide : vérifiez le champ « {fields[0]} »"
    return "Requête invalide : vérifiez les champs " + ", ".join(f"« {f} »" for f in fields)


def register_error_handlers(app: FastAPI, *, include_debug: bool) -> None:
    """Gestionnaires centralisés. `include_debug` = `settings.is_development` (jamais en production)."""

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        detail = _user_detail(exc.status_code, exc.detail)
        error_code = getattr(exc, "error_code", None) or default_error_code(exc.status_code)
        return error_response(
            request,
            exc.status_code,
            detail or _DEFAULT_DETAILS.get(exc.status_code, "Erreur de la requête"),
            error_code,
            headers=getattr(exc, "headers", None),
            # Un `detail` non textuel (liste, dict) n'est pas un message utilisateur : renvoyé en debug.
            debug=getattr(exc, "debug", None) or (None if isinstance(exc.detail, str) else exc.detail),
            include_debug=include_debug,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = list(exc.errors())
        return error_response(
            request, 422, validation_detail(errors), ErrorCode.VALIDATION_ERROR,
            debug={"errors": errors}, include_debug=include_debug,
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Filet de sécurité : une exception qu'aucune route n'a catchée (ex. crash natif d'une
        dépendance PDF) ne doit jamais atteindre le client comme un 500 sans corps ni trace."""
        logger.exception(
            "unhandled_exception",
            extra={"path": request.url.path, "request_id": current_request_id(request)},
        )
        return error_response(
            request, 500, INTERNAL_ERROR_DETAIL, ErrorCode.INTERNAL_ERROR,
            debug=exception_debug(exc), include_debug=include_debug,
        )
