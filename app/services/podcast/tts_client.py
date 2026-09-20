"""Client de synthèse vocale : interface TTSEngine + implémentation Piper (HTTP) avec cache disque.

Le moteur tourne dans un conteneur séparé (frontière IPC : aucun import du moteur GPL
dans ce code). Changer de moteur = fournir une autre implémentation de TTSEngine.
"""

import asyncio
import hashlib
import logging
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Protocol

import httpx

from app.core.config import Settings
from app.core.exceptions import TTSInvalidVoiceError, TTSUnavailableError

logger = logging.getLogger(__name__)


class TTSEngine(Protocol):
    name: str

    async def synthesize(self, text: str, voice: str) -> bytes:
        """Retourne le contenu d'un fichier WAV pour le texte et la voix donnés."""
        ...

    async def close(self) -> None: ...


class PiperHTTPEngine:
    """Moteur Piper via son serveur HTTP (POST / {"text", "voice"} → audio/wav)."""

    name = "piper"

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 60.0,
        max_retries: int = 3,
        client: httpx.AsyncClient | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._client = client or httpx.AsyncClient(base_url=base_url, timeout=timeout)
        self._max_retries = max(1, max_retries)
        self._sleep = sleep

    @classmethod
    def from_settings(cls, settings: Settings) -> "PiperHTTPEngine":
        return cls(
            settings.podcast_tts_base_url,
            timeout=settings.podcast_tts_timeout_seconds,
            max_retries=settings.podcast_tts_max_retries,
        )

    async def synthesize(self, text: str, voice: str) -> bytes:
        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = await self._client.post("/", json={"text": text, "voice": voice})
            except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = exc
                logger.warning("tts_call_failed", extra={"attempt": attempt + 1, "error": type(exc).__name__})
            else:
                if 400 <= response.status_code < 500:
                    raise TTSInvalidVoiceError(
                        f"Requête TTS rejetée ({response.status_code}) pour la voix « {voice} »"
                    )
                if response.status_code >= 500:
                    last_error = RuntimeError(f"HTTP {response.status_code}")
                    logger.warning("tts_server_error", extra={"attempt": attempt + 1, "status": response.status_code})
                elif not response.content.startswith(b"RIFF"):
                    raise TTSUnavailableError("Réponse TTS invalide : ce n'est pas un fichier WAV")
                else:
                    return response.content
            if attempt < self._max_retries - 1:
                await self._sleep(min(2**attempt, 10))
        raise TTSUnavailableError(
            f"Moteur TTS injoignable après {self._max_retries} tentatives"
        ) from last_error

    async def close(self) -> None:
        await self._client.aclose()


def cache_key(engine_name: str, voice: str, text: str) -> str:
    return hashlib.sha256(f"{engine_name}\x00{voice}\x00{text}".encode()).hexdigest()


async def synthesize_cached(engine: TTSEngine, cache_dir: Path, voice: str, text: str) -> Path:
    """Synthétise (ou réutilise) le WAV correspondant ; écriture atomique pour survivre à un crash."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{cache_key(engine.name, voice, text)}.wav"
    if target.exists() and target.stat().st_size > 0:
        return target
    audio = await engine.synthesize(text, voice)
    temporary = target.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_bytes(audio)
    os.replace(temporary, target)
    return target


async def synthesize_many(
    engine: TTSEngine,
    cache_dir: Path,
    items: list[tuple[str, str]],
    *,
    concurrency: int,
    on_progress: Callable[[int, int], Awaitable[None]] | None = None,
) -> list[Path]:
    """Synthétise une liste de (voix, texte) avec une concurrence bornée ; conserve l'ordre."""
    semaphore = asyncio.Semaphore(max(1, concurrency))
    done = 0

    async def one(voice: str, text: str) -> Path:
        nonlocal done
        async with semaphore:
            path = await synthesize_cached(engine, cache_dir, voice, text)
        done += 1
        if on_progress is not None:
            await on_progress(done, len(items))
        return path

    return list(await asyncio.gather(*(one(voice, text) for voice, text in items)))
