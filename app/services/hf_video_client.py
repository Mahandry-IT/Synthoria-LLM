"""HuggingFace Inference client for video generation.

Mirrors the retry/backoff pattern of gemini_client.py:
- Exponential backoff with jitter for retryable errors (429, 503).
- Timeout per call via asyncio.wait_for.
- Distinction between retryable and fatal errors.

Uses httpx (already in requirements.txt) — no new dependency.
"""

import asyncio
import logging
import random
from pathlib import Path

import httpx

from app.core.config import Settings
from app.core.exceptions import (
    HFVideoGenerationError,
    HFVideoRateLimitError,
    HFVideoServiceError,
    HFVideoUnavailableError,
)

logger = logging.getLogger(__name__)

_BACKOFF_BASE_SECONDS = 2.0
_BACKOFF_MAX_SECONDS = 60.0
_BACKOFF_JITTER = 0.25

# Max duration of a generated video clip (seconds) — used for poll timeout
_MAX_POLL_ATTEMPTS = 60
_POLL_INTERVAL_SECONDS = 5.0


def _is_retryable_status(status_code: int) -> bool:
    """429 (rate limit) and 503 (unavailable) are retryable."""
    return status_code in (429, 503)


class HFVideoClient:
    """Client for HuggingFace Inference API video generation.

    Supports two models with primary/fallback pattern.
    Video generation is asynchronous on HF side (queue-based):
    this client handles the polling loop internally.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._api_url = "https://api-inference.huggingface.co/models"
        self._headers = {}
        if settings.hf_api_token:
            self._headers["Authorization"] = f"Bearer {settings.hf_api_token}"

    def _ensure_configured(self) -> None:
        if not self._settings.hf_api_token:
            raise HFVideoUnavailableError("HF_API_TOKEN non configuré")

    async def _call_with_retry(
        self,
        payload: dict,
        model: str,
    ) -> bytes:
        """Appel HF Inference avec retry + backoff exponentiel.

        Stratégie :
          - 429/503 → retry avec backoff
          - Timeout → retry avec backoff
          - Autres erreurs → pas de retry

        Retour: bytes du fichier vidéo généré.
        """
        self._ensure_configured()
        url = f"{self._api_url}/{model}"
        last_error: Exception | None = None

        for attempt in range(self._settings.hf_video_max_retries):
            try:
                return await self._poll_generation(url, payload)
            except (HFVideoRateLimitError, HFVideoUnavailableError) as exc:
                last_error = exc
                backoff = min(
                    _BACKOFF_BASE_SECONDS ** (attempt + 1),
                    _BACKOFF_MAX_SECONDS,
                )
                jitter = backoff * _BACKOFF_JITTER * (2 * random.random() - 1)
                delay = max(0.1, backoff + jitter)
                logger.warning(
                    "hf_video_retry",
                    extra={
                        "attempt": attempt + 1,
                        "model": model,
                        "delay": round(delay, 2),
                        "error": str(exc),
                    },
                )
                if attempt < self._settings.hf_video_max_retries - 1:
                    await asyncio.sleep(delay)
            except HFVideoGenerationError:
                raise  # erreurs fatales (500 non retryable, etc.)

        raise last_error or HFVideoUnavailableError(
            f"HF Inference injoignable après {self._settings.hf_video_max_retries} tentatives"
        )

    async def _poll_generation(self, url: str, payload: dict) -> bytes:
        """Poll l'API HF Inference jusqu'à obtenir le résultat vidéo.

        L'API retourne un JSON avec `estimated_time` quand le modèle
        charge, puis les bytes vidéo quand c'est prêt.
        """
        async with httpx.AsyncClient(timeout=self._settings.hf_video_timeout_seconds) as client:
            for attempt in range(_MAX_POLL_ATTEMPTS):
                response = await client.post(
                    url,
                    json=payload,
                    headers=self._headers,
                )

                if response.status_code == 200:
                    content_type = response.headers.get("content-type", "")
                    if "video" in content_type or "octet-stream" in content_type:
                        return response.content
                    # JSON avec link (queued model)
                    try:
                        data = response.json()
                        if "video" in data:
                            video_url = data["video"]
                            video_response = await client.get(video_url)
                            if video_response.status_code == 200:
                                return video_response.content
                    except Exception:
                        pass
                    # Si on arrive ici, le 200 n'est pas un video stream
                    raise HFVideoGenerationError(
                        f"Réponse HF inattendue (status 200, content-type={content_type})",
                        error_code=200,
                    )

                if response.status_code == 503:
                    # Modèle en chargement — retry après estimated_time
                    try:
                        data = response.json()
                        estimated = data.get("estimated_time", _POLL_INTERVAL_SECONDS)
                        wait_time = min(estimated, 60.0)
                    except Exception:
                        wait_time = _POLL_INTERVAL_SECONDS
                    logger.info(
                        "hf_video_model_loading",
                        extra={"wait_seconds": wait_time, "attempt": attempt + 1},
                    )
                    await asyncio.sleep(wait_time)
                    continue

                if response.status_code == 429:
                    raise HFVideoRateLimitError(
                        "HF rate limit dépassé", error_code=429
                    )

                # Autre erreur
                raise HFVideoGenerationError(
                    f"Erreur HF Inference: {response.status_code} — {response.text[:200]}",
                    error_code=response.status_code,
                )

        raise HFVideoUnavailableError(
            "HF Inference timeout: polling trop long sans résultat"
        )

    async def generate_video(
        self,
        prompt: str,
        model: str | None = None,
    ) -> bytes:
        """Génère une vidéo à partir d'un prompt textuel.

        Paramètres:
            prompt: description visuelle de la scène (en anglais).
            model: modèle HF à utiliser (défaut: hf_video_model_primary).

        Retour: bytes du fichier vidéo MP4.

        Lève: HFVideoUnavailableError, HFVideoRateLimitError,
               HFVideoGenerationError.
        """
        target_model = model or self._settings.hf_video_model_primary
        payload = {"inputs": prompt}
        return await self._call_with_retry(payload, target_model)

    async def generate_video_with_fallback(
        self,
        prompt: str,
    ) -> tuple[bytes, str, bool]:
        """Essaie le modèle primary, puis fallback en cas d'échec.

        Retour: (video_bytes, model_used, fallback_used)
        """
        try:
            video = await self.generate_video(prompt, self._settings.hf_video_model_primary)
            return video, self._settings.hf_video_model_primary, False
        except (HFVideoGenerationError, HFVideoUnavailableError) as exc:
            logger.warning(
                "hf_video_primary_failed",
                extra={
                    "primary_model": self._settings.hf_video_model_primary,
                    "error": str(exc),
                },
            )
            # Fallback si le modèle primary est différent du fallback
            if self._settings.hf_video_model_primary != self._settings.hf_video_model_fallback:
                video = await self.generate_video(prompt, self._settings.hf_video_model_fallback)
                return video, self._settings.hf_video_model_fallback, True
            raise
