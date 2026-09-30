import asyncio
import json
import logging
import random
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from google import genai
from google.genai import types

from app.core.config import Settings
from app.core.exceptions import (
    GeminiDailyQuotaExceededError,
    GeminiInvalidResponseError,
    GeminiQuotaExceededError,
    GeminiServiceError,
    GeminiUnavailableError,
)
from app.services.gemini_rate_limit import GeminiRateLimiter
from app.services.quota_reset import next_pacific_midnight_utc

logger = logging.getLogger(__name__)

# Backoff exponentiel : base de 2s, max 60s, avec jitter ±25%
_BACKOFF_BASE_SECONDS = 2.0
_BACKOFF_MAX_SECONDS = 60.0
_BACKOFF_JITTER = 0.25

# Accumule les appels Gemini du contexte async courant (voir `GeminiClient.track_calls`) — `None`
# hors d'un `track_calls()`, ce qui ne doit jamais empêcher un appel Gemini de fonctionner.
_call_log: ContextVar[list[dict[str, Any]] | None] = ContextVar("gemini_call_log", default=None)


def _parse_gemini_error(exc: Exception) -> tuple[int | None, str | None, str | None]:
    """Extrait le code HTTP, le message d'erreur et le status depuis une exception Gemini.

    Retour: (error_code, error_status, error_message)
    """
    try:
        body = getattr(exc, "response", None)
        if body is None:
            return None, None, None
        data = getattr(body, "json", lambda: None)()
        if data is None:
            return None, None, None
        err = data.get("error", {})
        return err.get("code"), err.get("status"), err.get("message")
    except Exception:  # noqa: BLE001
        return None, None, None


def _is_rate_limited(exc: Exception) -> bool:
    """Détermine si l'erreur est un rate-limit 429 (retryable) vs une erreur fatale."""
    code, status, _ = _parse_gemini_error(exc)
    if code == 429:
        return True
    if status and "RESOURCE_EXHAUSTED" in status.upper():
        return True
    # Fallback string matching pour les erreurs non structurées
    message = str(exc).lower()
    return "429" in message or "resource_exhausted" in message or "rate limit" in message


def _extract_retry_delay(exc: Exception) -> float | None:
    """Extrait le retryDelay de la réponse d'erreur Google API, si disponible."""
    try:
        error_body = getattr(exc, "response", None)
        if error_body is None:
            return None
        body = getattr(error_body, "json", lambda: None)()
        if body is None:
            return None
        for detail in body.get("error", {}).get("details", []):
            if detail.get("@type", "").endswith("RetryInfo"):
                delay_str = detail.get("retryDelay", "")
                if delay_str.endswith("s"):
                    return float(delay_str[:-1])
    except Exception:  # noqa: BLE001
        pass
    return None


def _classify_quota_error(exc: Exception) -> str:
    """Distingue un quota JOURNALIER (aucun retry ne peut réussir avant le reset) d'un simple
    rate-limit MINUTE (transitoire, `_extract_retry_delay` suffit) à partir de
    `error.details[].quotaId` (structure `QuotaFailure` de l'API Google).

    Retour: `"day"`, `"minute"`, ou `"unknown"` si la violation n'est pas identifiable.

    Le format exact de `quotaId` n'a pas été confirmé sur un vrai 429 de ce projet — tant que ce
    n'est pas vérifié en conditions réelles, `"unknown"` reste traité comme retryable (voir
    `_call_with_retry`) : ne jamais fail-fast sur une classification incertaine.
    """
    try:
        body = getattr(exc, "response", None)
        if body is None:
            return "unknown"
        data = getattr(body, "json", lambda: None)()
        if data is None:
            return "unknown"
        for detail in data.get("error", {}).get("details", []):
            if not detail.get("@type", "").endswith("QuotaFailure"):
                continue
            for violation in detail.get("violations", []):
                quota_id = violation.get("quotaId", "")
                if "PerDay" in quota_id:
                    return "day"
                if "PerMinute" in quota_id:
                    return "minute"
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


def _strip_additional_properties(schema: dict) -> dict:
    """Nettoie récursivement le JSON schema pour compatibilité Gemini API.

    Gemini rejette `additionalProperties` dans le payload. Pydantic v2 l'ajoute
    par défaut sur les dict et les modèles. Cette fonction le supprime partout.
    """
    cleaned: dict = {}
    for key, value in schema.items():
        if key == "additionalProperties":
            continue
        if isinstance(value, dict):
            cleaned[key] = _strip_additional_properties(value)
        elif isinstance(value, list):
            cleaned[key] = [
                _strip_additional_properties(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            cleaned[key] = value
    return cleaned


class GeminiClient:
    """Encapsule les appels Gemini pour la génération de cours (retry + timeout inclus).

    Contrainte API (Gemini Flash) : `google_search` (grounding) et `response_schema`
    ne peuvent pas être combinés dans un même appel. Ce client expose donc deux
    méthodes distinctes, à enchaîner côté orchestration (voir `course_generator.py`) :
      1. `search_grounded`   — réponse libre, avec recherche web (pas de schema).
      2. `format_structured` — reformatage strict en JSON (pas de recherche web).
    """

    def __init__(self, settings: Settings, client: "genai.Client | None" = None) -> None:
        self._settings = settings
        self._client = client or (
            genai.Client(api_key=settings.gemini_api_key) if settings.gemini_api_key else None
        )
        # Une instance par processus (voir gemini_rate_limit.py) : partagée entre tous les appels
        # tant que `GeminiClient` reste un singleton applicatif (app.state.gemini_client).
        self._rate_limiter = GeminiRateLimiter(settings.gemini_rpm_limit)

    def _ensure_configured(self) -> None:
        if self._client is None:
            raise GeminiUnavailableError("GEMINI_API_KEY non configurée")

    @property
    def is_configured(self) -> bool:
        return self._client is not None

    def _log_call(self, *, model: str, method: str, duration_ms: float, status: str, attempt: int) -> None:
        """Log structuré `gemini_call` (jamais le prompt ni la clé) + accumulation dans
        `_call_log` si un `track_calls()` est actif (sinon no-op, voir `_call_log`)."""
        logger.info(
            "gemini_call",
            extra={"model": model, "method": method, "duration_ms": round(duration_ms, 1), "status": status, "attempt": attempt},
        )
        log = _call_log.get()
        if log is not None:
            log.append({"model": model, "method": method, "duration_ms": round(duration_ms, 1), "status": status})

    @contextmanager
    def track_calls(self):
        """Accumule et logue en un `gemini_calls_summary` tous les appels Gemini faits pendant
        le bloc (génération d'un cours entier, typiquement) — diagnostic de consommation de quota."""
        token = _call_log.set([])
        try:
            yield
        finally:
            calls = _call_log.get() or []
            by_model: dict[str, int] = {}
            by_method: dict[str, int] = {}
            for call in calls:
                by_model[call["model"]] = by_model.get(call["model"], 0) + 1
                by_method[call["method"]] = by_method.get(call["method"], 0) + 1
            logger.info(
                "gemini_calls_summary",
                extra={"total_calls": len(calls), "by_model": by_model, "by_method": by_method},
            )
            _call_log.reset(token)

    async def _call_with_retry(self, func: Any, *args: Any, method: str = "", **kwargs: Any) -> Any:
        """Retry avec backoff exponentiel + jitter.

        Stratégie :
          - 404 (modèle introuvable) → échec immédiat, aucun retry (`GeminiUnavailableError`).
          - 429 rate-limited, quota JOURNALIER identifié → échec immédiat, aucun retry
            (`GeminiDailyQuotaExceededError`, voir `_classify_quota_error`) : aucune tentative
            supplémentaire ne peut réussir avant le prochain reset.
          - 429 rate-limited, quota minute ou non classifiable → retry avec backoff (extrait
            `retryDelay` si dispo).
          - Timeout / autres erreurs → retry avec backoff standard.

        `method` (nom de la méthode publique appelante, ex. "format_structured") sert uniquement
        au log `gemini_call`/`gemini_calls_summary` — jamais au comportement de retry.
        """
        last_error: Exception | None = None
        rate_limit_delay: float | None = None
        model = args[0] if args and isinstance(args[0], str) else "unknown"

        for attempt in range(self._settings.gemini_max_retries):
            started = time.monotonic()
            try:
                # Espace les appels (y compris les retries) pour rester sous GEMINI_RPM_LIMIT
                # plutôt que de laisser Google renvoyer 429 puis retenter après coup.
                await self._rate_limiter.acquire()
                result = await asyncio.wait_for(
                    asyncio.to_thread(func, *args, **kwargs),
                    timeout=self._settings.gemini_timeout_seconds,
                )
                self._log_call(
                    model=model, method=method, duration_ms=(time.monotonic() - started) * 1000,
                    status="ok", attempt=attempt + 1,
                )
                return result
            except TimeoutError as exc:
                last_error = exc
                self._log_call(
                    model=model, method=method, duration_ms=(time.monotonic() - started) * 1000,
                    status="timeout", attempt=attempt + 1,
                )
                logger.warning("gemini_call_timeout", extra={"attempt": attempt + 1})
            except Exception as exc:  # noqa: BLE001 - le SDK ne type pas finement ses erreurs
                self._log_call(
                    model=model, method=method, duration_ms=(time.monotonic() - started) * 1000,
                    status="error", attempt=attempt + 1,
                )
                error_code, _, error_message = _parse_gemini_error(exc)
                if error_code == 404:
                    # Modèle inconnu/désactivé : jamais résolu en réessayant le même modèle.
                    raise GeminiUnavailableError(
                        f"Modèle Gemini introuvable (404): {error_message or exc}",
                        error_code=404, error_message=error_message,
                    ) from exc
                if _is_rate_limited(exc) and _classify_quota_error(exc) == "day":
                    # Quota JOURNALIER : mathématiquement impossible de réussir avant le reset,
                    # retenter ne ferait que gaspiller du RPM partagé avec l'autre process (api/worker).
                    retry_at = next_pacific_midnight_utc()
                    logger.warning("gemini_daily_quota_exceeded", extra={"model": model, "retry_at": retry_at.isoformat()})
                    raise GeminiDailyQuotaExceededError(
                        f"Quota Gemini journalier dépassé: {error_message or exc}",
                        error_code=error_code, error_message=error_message, retry_at=retry_at,
                    ) from exc
                if _is_rate_limited(exc):
                    last_error = exc
                    rate_limit_delay = _extract_retry_delay(exc)
                    backoff = (
                        min(rate_limit_delay, _BACKOFF_MAX_SECONDS)
                        if rate_limit_delay
                        else min(_BACKOFF_BASE_SECONDS ** (attempt + 1), _BACKOFF_MAX_SECONDS)
                    )
                    jitter = backoff * _BACKOFF_JITTER * (2 * random.random() - 1)
                    delay = max(0.1, backoff + jitter)
                    logger.warning(
                        "gemini_rate_limited",
                        extra={"attempt": attempt + 1, "delay": round(delay, 2), "retry_after": rate_limit_delay},
                    )
                else:
                    last_error = exc
                    logger.warning(
                        "gemini_call_failed", extra={"attempt": attempt + 1, "error": str(exc)}
                    )

            if attempt < self._settings.gemini_max_retries - 1:
                if last_error and _is_rate_limited(last_error):
                    backoff = (
                        min(rate_limit_delay, _BACKOFF_MAX_SECONDS)
                        if rate_limit_delay
                        else min(_BACKOFF_BASE_SECONDS ** (attempt + 1), _BACKOFF_MAX_SECONDS)
                    )
                    jitter = backoff * _BACKOFF_JITTER * (2 * random.random() - 1)
                    await asyncio.sleep(max(0.1, backoff + jitter))
                else:
                    await asyncio.sleep(min(_BACKOFF_BASE_SECONDS ** (attempt + 1), _BACKOFF_MAX_SECONDS))

        # Extraire les infos structurées de la dernière erreur
        error_code, error_status, error_message = _parse_gemini_error(last_error) if last_error else (None, None, None)
        detail = error_message or str(last_error) if last_error else "unknown error"

        if error_code == 429 or (error_status and "RESOURCE_EXHAUSTED" in error_status.upper()):
            raise GeminiQuotaExceededError(
                f"Quota Gemini dépassé: {detail}",
                error_code=error_code,
                error_message=detail,
            ) from last_error

        raise GeminiUnavailableError(
            f"Gemini injoignable après {self._settings.gemini_max_retries} tentatives: {detail}",
            error_code=error_code,
            error_message=detail,
        ) from last_error

    async def _call_with_model_cascade(self, build_request: Any, models: list[str], *, method: str = "") -> Any:
        """Essaie chaque modèle de `models` dans l'ordre, bascule sur le suivant
        si l'appel échoue avec une erreur Gemini non transitoire côté appelant
        (`GeminiQuotaExceededError`/`GeminiUnavailableError` — `_call_with_retry`
        ne laisse jamais fuiter d'exception brute, ces deux types couvrent donc
        déjà tous les cas : quota atteint, service indisponible, ou toute autre
        erreur non retryable après épuisement des tentatives).

        Paramètres:
            build_request: callable `(model: str) -> Any` construisant l'appel
                Gemini pour un modèle donné (passé à `_call_with_retry`).
            models: liste ordonnée de modèles à essayer (ex. [flash, flash_lite]).

        Retour: la réponse du premier modèle qui réussit.

        Lève: l'erreur du dernier modèle essayé, si tous échouent.
        """
        last_error: GeminiServiceError | None = None
        for i, model in enumerate(models):
            try:
                return await self._call_with_retry(build_request, model, method=method)
            except (GeminiQuotaExceededError, GeminiUnavailableError) as exc:
                last_error = exc
                next_model = models[i + 1] if i + 1 < len(models) else None
                if next_model is not None:
                    logger.warning(
                        "gemini_model_fallback",
                        extra={"failed_model": model, "next_model": next_model, "error": type(exc).__name__},
                    )
        raise last_error  # type: ignore[misc]

    async def search_grounded(self, prompt: str, system_instruction: str) -> tuple[str, list[dict]]:
        """
        Appel 1 : génère une réponse groundée par recherche web (sans response_schema).

        Paramètres:
            prompt: contenu utilisateur (question + contexte RAG éventuel).
            system_instruction: prompt système (méthode pédagogique What/Why/How).

        Retour: tuple (texte_brut, sources_web) où chaque source web est
            {"type": "web", "label": str, "reference": str}.

        Fonctionnement: essaie `gemini_model_flash` puis, en cas de quota
        dépassé ou d'indisponibilité, bascule sur `gemini_model_flash_lite`
        (voir `_call_with_model_cascade`).

        Lève: GeminiUnavailableError, GeminiQuotaExceededError (du dernier
            modèle essayé, si les deux échouent).
        """
        self._ensure_configured()

        def _run(model: str) -> Any:
            return self._client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                ),
            )

        response = await self._call_with_model_cascade(
            _run, [self._settings.gemini_model_flash, self._settings.gemini_model_flash_lite],
            method="search_grounded",
        )
        text = getattr(response, "text", "") or ""
        return text, self._extract_web_sources(response)

    def _extract_web_sources(self, response: Any) -> list[dict]:
        """Extrait les sources web du grounding_metadata renvoyé par Gemini. Ne bloque jamais."""
        web_sources: list[dict] = []
        try:
            for candidate in getattr(response, "candidates", None) or []:
                grounding = getattr(candidate, "grounding_metadata", None)
                for chunk in getattr(grounding, "grounding_chunks", None) or []:
                    web = getattr(chunk, "web", None)
                    if web is None:
                        continue
                    web_sources.append(
                        {
                            "type": "web",
                            "label": getattr(web, "title", "") or getattr(web, "uri", ""),
                            "reference": getattr(web, "uri", ""),
                        }
                    )
        except Exception as exc:  # pragma: no cover - metadata optionnelle
            logger.warning("gemini_grounding_metadata_parse_failed: %s", exc)
        return web_sources

    async def reformulate_query(self, query: str, filename: str | list[str] | None = None) -> str:
        """Reformule une question vague en une requête précise pour la recherche vectorielle.

        Stratégie : un seul appel au modèle lite, réponse JSON simple.
        En cas d'échec, retourne la query originale (pas de blocage).
        """
        self._ensure_configured()

        if filename is None:
            context_hint = ""
        elif isinstance(filename, list):
            context_hint = f" Les documents s'intitulent: {', '.join(filename)}."
        else:
            context_hint = f" Le document s'intitule: {filename}."

        prompt = (
            f"Reformule cette question en une requête de recherche précise et technique "
            f"pour trouver le contenu pertinent dans un cours. "
            f"Réponds UNIQUEMENT avec le JSON: {{\"query\": \"...\"}}. "
            f"Pas d'explication, pas de texte hors JSON."
            f"{context_hint}\n\n"
            f"Question: {query}"
        )

        def _run(model: str) -> Any:
            return self._client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )

        try:
            response = await self._call_with_retry(
                _run, self._settings.gemini_model_flash_lite, method="reformulate_query"
            )
            text = getattr(response, "text", "") or ""
            data = json.loads(text)
            reformulated = data.get("query", "")
            if reformulated and len(reformulated) > 5:
                logger.info(
                    "query_reformulated",
                    extra={"original": query[:80], "reformulated": reformulated[:80]},
                )
                return reformulated
        except Exception as exc:  # noqa: BLE001 - fallback silencieux
            logger.warning("query_reformulation_failed", extra={"error": str(exc)})

        return query

    async def format_structured(
        self, raw_answer: str, system_instruction: str, *, response_schema: Any | None = None
    ) -> dict:
        """
        Reformate une réponse brute en JSON structuré strict (sans google_search).

        Si response_schema est une classe Pydantic, extrait le JSON schema via
        model_json_schema(). Utilise response_json_schema (bypass validation SDK).

        Stratégie : tente d'abord avec le modèle lite, fallback sur le modèle flash si son quota
        est épuisé (429, quota séparé de celui de flash) ou si le schema est trop complexe pour
        lui (400 InvalidArgument).
        """
        self._ensure_configured()
        if response_schema is not None:
            clean_schema = _strip_additional_properties(
                response_schema.model_json_schema() if hasattr(response_schema, "model_json_schema") else response_schema
            )
        else:
            raise GeminiUnavailableError("Aucun response_schema fourni à format_structured")

        def _run(model: str) -> Any:
            return self._client.models.generate_content(
                model=model,
                contents=raw_answer,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_json_schema=clean_schema,
                ),
            )

        # Essai avec le modèle lite (moins cher)
        try:
            response = await self._call_with_retry(
                _run, self._settings.gemini_model_flash_lite, method="format_structured"
            )
        except GeminiQuotaExceededError:
            # Quota du lite épuisé : flash a son propre quota séparé, distinct de celui du lite
            # (contrairement au cas 400 ci-dessous, jamais résolu en re-tentant le même modèle).
            logger.info("gemini_lite_quota_fallback_to_flash")
            response = await self._call_with_retry(
                _run, self._settings.gemini_model_flash, method="format_structured"
            )
        except GeminiUnavailableError as exc:
            # Si le lite échoue avec un 400 (schema trop complexe), retry avec flash
            error_msg = str(exc).lower()
            if "400" in error_msg or "invalid" in error_msg or "bad request" in error_msg:
                logger.info("gemini_lite_schema_fallback_to_flash")
                response = await self._call_with_retry(
                    _run, self._settings.gemini_model_flash, method="format_structured"
                )
            else:
                raise

        text = getattr(response, "text", "") or ""
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise GeminiInvalidResponseError(f"Réponse Gemini non-JSON: {exc}") from exc

    async def describe_image(self, image_bytes: bytes, mime_type: str, system_instruction: str) -> str:
        """Décrit une image (ex. figure extraite d'un PDF) via `gemini_model_flash_lite`.

        Passe par le même rate limiter / retry-backoff / timeout que les autres appels Gemini
        (`_call_with_retry`) — contrairement à un appel SDK direct qui pourrait déclencher une
        rafale de requêtes non espacées et épuiser le quota (ex. plusieurs images par PDF ingéré).
        """
        self._ensure_configured()

        def _run(model: str) -> Any:
            return self._client.models.generate_content(
                model=model,
                contents=[types.Part.from_bytes(data=image_bytes, mime_type=mime_type)],
                config=types.GenerateContentConfig(system_instruction=system_instruction),
            )

        response = await self._call_with_retry(
            _run, self._settings.gemini_model_flash_lite, method="describe_image"
        )
        return getattr(response, "text", "") or ""

    async def rank_images(
        self, image_bytes_list: list[tuple[bytes, str]], prompt: str, *, system_instruction: str, response_schema: Any
    ) -> dict:
        """Appel structuré multimodal (images + texte) : vérifie la pertinence de candidats d'image.

        `image_bytes_list` : une entrée `(bytes, mime_type)` par candidat numéroté dans `prompt`
        (même ordre). `gemini_model_flash_lite` uniquement — jamais flash, ce jugement de
        pertinence ne justifie pas le coût du modèle complet. Passe par `_call_with_retry` comme
        tout appel Gemini (rate limiter, retry-backoff, timeout).
        """
        self._ensure_configured()
        clean_schema = _strip_additional_properties(
            response_schema.model_json_schema() if hasattr(response_schema, "model_json_schema") else response_schema
        )
        parts: list[Any] = [
            types.Part.from_bytes(data=data, mime_type=mime) for data, mime in image_bytes_list
        ]
        parts.append(prompt)

        def _run(model: str) -> Any:
            return self._client.models.generate_content(
                model=model,
                contents=parts,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_json_schema=clean_schema,
                ),
            )

        response = await self._call_with_retry(
            _run, self._settings.gemini_model_flash_lite, method="rank_images"
        )
        text = getattr(response, "text", "") or ""
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise GeminiInvalidResponseError(f"Réponse Gemini non-JSON: {exc}") from exc