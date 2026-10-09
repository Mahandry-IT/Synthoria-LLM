"""Identifiant de corrélation par requête (`X-Request-ID`), repris du client ou généré."""

import logging
import re
import uuid
from contextvars import ContextVar

from starlette.types import ASGIApp, Message, Receive, Scope, Send

REQUEST_ID_HEADER = "X-Request-ID"
_HEADER_KEY = REQUEST_ID_HEADER.lower().encode("latin-1")
# Identifiant client accepté tel quel seulement s'il est court et sans caractère de contrôle :
# il est recopié dans les logs et en en-tête de réponse (injection de logs / d'en-têtes).
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_NO_REQUEST_ID = "-"

request_id_var: ContextVar[str] = ContextVar("request_id", default=_NO_REQUEST_ID)


def _incoming_request_id(scope: Scope) -> str | None:
    for key, value in scope.get("headers", []):
        if key == _HEADER_KEY:
            candidate = value.decode("latin-1").strip()
            return candidate if _VALID_REQUEST_ID.match(candidate) else None
    return None


class RequestIdMiddleware:
    """Middleware ASGI pur (pas `BaseHTTPMiddleware`) : expose l'identifiant dans
    `request.state.request_id` (lu par les gestionnaires d'erreurs, y compris le 500 global qui
    s'exécute hors de ce middleware), dans `request_id_var` (logs) et dans l'en-tête de réponse."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _incoming_request_id(scope) or uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        request_id_var.set(request_id)

        async def send_with_header(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = [(k, v) for k, v in message.get("headers", []) if k.lower() != _HEADER_KEY]
                headers.append((_HEADER_KEY, request_id.encode("latin-1")))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_header)


class RequestIdLogFilter(logging.Filter):
    """Ajoute `record.request_id` (« - » hors requête : worker, démarrage) pour le format de log."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = request_id_var.get()
        return True
