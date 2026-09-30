import asyncio
import hashlib
import json
import logging
import random
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from google import genai
from google.genai import types
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.core.config import Settings
from app.core.exceptions import (
    GeminiDailyQuotaExceededError,
    GeminiInvalidResponseError,
    GeminiQuotaExceededError,
    GeminiServiceError,
    GeminiUnavailableError,
)
from app.repositories.gemini_response_cache_repository import compute_query_hash
from app.services.gemini_quota_manager import GeminiQuotaManager
from app.services.gemini_rate_limit import GeminiRateLimiter
from app.services.gemini_response_cache_manager import GeminiResponseCacheManager
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

    def __init__(
        self,
        settings: Settings,
        client: "genai.Client | None" = None,
        session_factory: async_sessionmaker | None = None,
    ) -> None:
        self._settings = settings
        self._client = client or (
            genai.Client(api_key=settings.gemini_api_key) if settings.gemini_api_key else None
        )
        # Une instance par processus (voir gemini_rate_limit.py) : partagée entre tous les appels
        # tant que `GeminiClient` reste un singleton applicatif (app.state.gemini_client).
        # `gemini_rpm_share` répartit le RPM partagé entre api/worker (ex. 0.7/0.3) plutôt que de
        # laisser chacun croire qu'il dispose de la totalité du quota.
        self._rate_limiter = GeminiRateLimiter(max(1, round(settings.gemini_rpm_limit * settings.gemini_rpm_share)))
        # `session_factory=None` : GeminiQuotaManager devient un no-op silencieux (toujours
        # disponible) — utilisable sans DB (scripts, tests) sans changer le comportement.
        self._quota_manager = GeminiQuotaManager(settings, session_factory)
        # Idem pour le cache de réponses : `session_factory=None` → toujours cache miss.
        self._response_cache = GeminiResponseCacheManager(settings, session_factory)

    def _ensure_configured(self) -> None:
        if self._client is None:
            raise GeminiUnavailableError("GEMINI_API_KEY non configurée")

    @property
    def is_configured(self) -> bool:
        return self._client is not None

    async def is_degraded(self) -> bool:
        """True si même le modèle le plus robuste configuré est marqué indisponible (voir
        `GeminiQuotaManager.is_degraded`) : les appelants avec des appels Gemini OPTIONNELS
        (best-effort, ex. complétion de couverture, classement de vidéos) devraient les sauter
        plutôt que les tenter en vain."""
        return await self._quota_manager.is_degraded()

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

        if not await self._quota_manager.is_available(model):
            # Déjà marqué épuisé (429 jour reçu plus tôt, ou budget RPD configuré atteint) :
            # aucun appel réseau. `_call_with_model_cascade` catche cette erreur pour passer au
            # modèle suivant, comme n'importe quel autre échec non transitoire.
            raise GeminiDailyQuotaExceededError(
                f"Modèle Gemini {model} marqué indisponible (quota épuisé)",
                retry_at=next_pacific_midnight_utc(),
            )

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
                await self._quota_manager.record_success(model)
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
                    await self._quota_manager.record_daily_exhausted(model)
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

        Fonctionnement: essaie les modèles de `settings.gemini_chain_search` dans l'ordre (voir
        `_call_with_model_cascade`), en sautant ceux marqués indisponibles par le disjoncteur de
        quota (`GeminiQuotaManager`).

        Lève: GeminiUnavailableError, GeminiQuotaExceededError (du dernier
            modèle essayé, si tous échouent).
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
            _run, self._settings.gemini_chain_search, method="search_grounded",
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

        Stratégie : `settings.gemini_chain_light` (lite d'abord), réponse JSON simple. Résultat mis
        en cache (`gemini_response_cache`, TTL `gemini_response_cache_ttl_hours`) par hash du
        prompt complet — la même question posée deux fois (même fichier) évite un second appel.
        En cas d'échec (tous les modèles de la chaîne), retourne la query originale (pas de blocage).
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
        query_hash = compute_query_hash("reformulate_query", prompt)

        cached = await self._response_cache.get(query_hash)
        if cached is not None:
            reformulated = cached.get("query", "")
            if reformulated:
                return reformulated

        def _run(model: str) -> Any:
            return self._client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )

        try:
            response = await self._call_with_model_cascade(
                _run, self._settings.gemini_chain_light, method="reformulate_query"
            )
            text = getattr(response, "text", "") or ""
            data = json.loads(text)
            reformulated = data.get("query", "")
            if reformulated and len(reformulated) > 5:
                logger.info(
                    "query_reformulated",
                    extra={"original": query[:80], "reformulated": reformulated[:80]},
                )
                await self._response_cache.set(query_hash, "reformulate_query", {"query": reformulated})
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

        Stratégie : `settings.gemini_chain_generation` (lite d'abord, moins cher), fallback sur le
        modèle suivant si le courant est en quota ou indisponible (`_call_with_model_cascade`).
        Un 400 (schema trop complexe) sur le lite est aussi couvert : `_call_with_retry` le classe
        en `GeminiUnavailableError`, que le cascade rattrape comme n'importe quel autre échec.
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

        response = await self._call_with_model_cascade(
            _run, self._settings.gemini_chain_generation, method="format_structured"
        )

        text = getattr(response, "text", "") or ""
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise GeminiInvalidResponseError(f"Réponse Gemini non-JSON: {exc}") from exc

    async def _call_multimodal_structured(
        self,
        image_bytes_list: list[tuple[bytes, str]],
        prompt: str,
        *,
        system_instruction: str,
        response_schema: Any,
        chain: list[str],
        method: str,
    ) -> dict:
        """Appel structuré multimodal (N images numérotées + texte) partagé par `rank_images` et
        `describe_images` : un seul appel Gemini plutôt qu'un par image, `response_schema` impose
        le format JSON de sortie. Passe par `_call_with_model_cascade`/`_call_with_retry` comme
        tout appel Gemini (rate limiter, retry-backoff, timeout, disjoncteur de quota).

        Résultat mis en cache (`gemini_response_cache`) par hash de la méthode + du prompt + de
        l'instruction système + du contenu des images (sha256 par image, jamais les bytes
        eux-mêmes en clé) + du schema de sortie complet (pas seulement son nom : un changement de
        champ sans renommage de classe doit aussi invalider le cache) — un même lot d'images déjà
        traité (ex. PDF réingéré) évite un nouvel appel.
        """
        self._ensure_configured()
        clean_schema = _strip_additional_properties(
            response_schema.model_json_schema() if hasattr(response_schema, "model_json_schema") else response_schema
        )
        schema_fingerprint = json.dumps(clean_schema, sort_keys=True)
        image_hashes = [hashlib.sha256(data).hexdigest() for data, _ in image_bytes_list]
        query_hash = compute_query_hash(method, prompt, system_instruction, schema_fingerprint, *image_hashes)

        cached = await self._response_cache.get(query_hash)
        if cached is not None:
            return cached

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

        response = await self._call_with_model_cascade(_run, chain, method=method)
        text = getattr(response, "text", "") or ""
        try:
            result = json.loads(text)
        except json.JSONDecodeError as exc:
            raise GeminiInvalidResponseError(f"Réponse Gemini non-JSON: {exc}") from exc

        await self._response_cache.set(query_hash, method, result)
        return result

    async def rank_images(
        self, image_bytes_list: list[tuple[bytes, str]], prompt: str, *, system_instruction: str, response_schema: Any
    ) -> dict:
        """Vérifie la pertinence de candidats d'image (une seule sélectionnée).

        `image_bytes_list` : une entrée `(bytes, mime_type)` par candidat numéroté dans `prompt`
        (même ordre). `settings.gemini_chain_light` (lite d'abord — ce jugement de pertinence ne
        justifie pas le coût du modèle complet en temps normal ; le cascade n'y bascule qu'en
        dernier recours, si le lite est indisponible, plutôt que de renoncer entièrement au
        classement).
        """
        return await self._call_multimodal_structured(
            image_bytes_list, prompt,
            system_instruction=system_instruction, response_schema=response_schema,
            chain=self._settings.gemini_chain_light, method="rank_images",
        )

    async def describe_images(
        self, image_bytes_list: list[tuple[bytes, str]], prompt: str, *, system_instruction: str, response_schema: Any
    ) -> dict:
        """Décrit N images (ex. figures extraites d'un PDF) en un seul appel Gemini au lieu d'un
        appel par image — réduit la consommation de quota proportionnellement au nombre d'images
        regroupées (voir `gemini_vision.py`).

        `image_bytes_list` : une entrée `(bytes, mime_type)` par image numérotée dans `prompt`
        (même ordre). `settings.gemini_chain_light` (lite d'abord).
        """
        return await self._call_multimodal_structured(
            image_bytes_list, prompt,
            system_instruction=system_instruction, response_schema=response_schema,
            chain=self._settings.gemini_chain_light, method="describe_images",
        )