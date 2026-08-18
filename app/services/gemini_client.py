import asyncio
import json
import logging
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
        last_error: Exception | None = None

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
                message = str(exc).lower()
                if "quota" in message or "rate limit" in message or "429" in message:
                    raise GeminiQuotaExceededError(f"Quota Gemini dépassé: {exc}") from exc
                last_error = exc
                logger.warning(
                    "gemini_call_failed", extra={"attempt": attempt + 1, "error": str(exc)}
                )

            if attempt < self._settings.gemini_max_retries - 1:
                await asyncio.sleep(min(2**attempt, 10))

        raise GeminiUnavailableError(
            f"Gemini injoignable après {self._settings.gemini_max_retries} tentatives"
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
        Appel 2 : reformate une réponse brute en JSON structuré strict (sans google_search).

        Paramètres:
            raw_answer: texte à structurer (réponse brute de `search_grounded` + contexte sources).
            response_schema: modèle Pydantic (ou schéma JSON) décrivant la structure attendue.
            system_instruction: prompt système de formatage.

        Retour: dict JSON validé syntaxiquement (validation métier faite en aval par Pydantic).

        Lève: GeminiUnavailableError, GeminiQuotaExceededError, GeminiInvalidResponseError.
        """
        self._ensure_configured()

        def _run() -> Any:
            return self._client.models.generate_content(
                model=self._settings.gemini_model_flash_lite,
                contents=raw_answer,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=response_schema,
                ),
            )

        response = await self._call_with_retry(_run)
        text = getattr(response, "text", "") or ""
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise GeminiInvalidResponseError(f"Réponse Gemini non-JSON: {exc}") from exc