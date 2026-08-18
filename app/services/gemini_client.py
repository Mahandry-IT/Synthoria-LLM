import asyncio
import json
import logging
import random
from typing import Any

from google import genai
from google.genai import types

from app.core.config import Settings
from app.core.exceptions import (
    GeminiInvalidResponseError,
    GeminiQuotaExceededError,
    GeminiUnavailableError,
)

logger = logging.getLogger(__name__)

# Backoff exponentiel : base de 2s, max 60s, avec jitter ±25%
_BACKOFF_BASE_SECONDS = 2.0
_BACKOFF_MAX_SECONDS = 60.0
_BACKOFF_JITTER = 0.25


def _is_rate_limited(exc: Exception) -> bool:
    """Détermine si l'erreur est un rate-limit 429 (retryable) vs une erreur fatale."""
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

    def _ensure_configured(self) -> None:
        if self._client is None:
            raise GeminiUnavailableError("GEMINI_API_KEY non configurée")

    async def _call_with_retry(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Retry avec backoff exponentiel + jitter. Les 429 sont retryables.

        Stratégie :
          - Timeout / erreurs réseau → retry avec backoff
          - 429 rate-limited → retry avec backoff (extrait retryDelay si dispo)
          - Autres erreurs (clé invalide, etc.) → pas de retry
        """
        last_error: Exception | None = None
        rate_limit_delay: float | None = None

        for attempt in range(self._settings.gemini_max_retries):
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(func, *args, **kwargs),
                    timeout=self._settings.gemini_timeout_seconds,
                )
            except TimeoutError as exc:
                last_error = exc
                logger.warning("gemini_call_timeout", extra={"attempt": attempt + 1})
            except Exception as exc:  # noqa: BLE001 - le SDK ne type pas finement ses erreurs
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

        detail = str(last_error) if last_error else "unknown error"
        raise GeminiUnavailableError(
            f"Gemini injoignable après {self._settings.gemini_max_retries} tentatives: {detail}"
        ) from last_error

    async def search_grounded(self, prompt: str, system_instruction: str) -> tuple[str, list[dict]]:
        """
        Appel 1 : génère une réponse groundée par recherche web (sans response_schema).

        Paramètres:
            prompt: contenu utilisateur (question + contexte RAG éventuel).
            system_instruction: prompt système (méthode pédagogique What/Why/How).

        Retour: tuple (texte_brut, sources_web) où chaque source web est
            {"type": "web", "label": str, "reference": str}.

        Lève: GeminiUnavailableError, GeminiQuotaExceededError.
        """
        self._ensure_configured()

        def _run() -> Any:
            return self._client.models.generate_content(
                model=self._settings.gemini_model_flash,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                ),
            )

        response = await self._call_with_retry(_run)
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

    async def format_structured(
        self, raw_answer: str, response_schema: Any, system_instruction: str
    ) -> dict:
        """
        Reformate une réponse brute en JSON structuré strict (sans google_search).

        Stratégie : tente d'abord avec le modèle lite, fallback sur le modèle flash
        si le schema est trop complexe (400 InvalidArgument).
        """
        self._ensure_configured()
        clean_schema = _strip_additional_properties(
            response_schema.model_json_schema() if hasattr(response_schema, "model_json_schema") else response_schema
        )

        def _run(model: str) -> Any:
            return self._client.models.generate_content(
                model=model,
                contents=raw_answer,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=clean_schema,
                ),
            )

        # Essai avec le modèle lite (moins cher)
        try:
            response = await self._call_with_retry(_run, self._settings.gemini_model_flash_lite)
        except GeminiUnavailableError as exc:
            # Si le lite échoue avec un 400 (schema trop complexe), retry avec flash
            error_msg = str(exc).lower()
            if "400" in error_msg or "invalid" in error_msg or "bad request" in error_msg:
                logger.info("gemini_lite_schema_fallback_to_flash")
                response = await self._call_with_retry(_run, self._settings.gemini_model_flash)
            else:
                raise

        text = getattr(response, "text", "") or ""
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise GeminiInvalidResponseError(f"Réponse Gemini non-JSON: {exc}") from exc