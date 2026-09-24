class OllamaServiceError(Exception):
    """Erreur générique lors de la communication avec Ollama."""


class OllamaUnavailableError(OllamaServiceError):
    """Ollama est injoignable (connexion, timeout)."""


class OllamaModelNotFoundError(OllamaServiceError):
    """Le modèle demandé n'existe pas sur l'instance Ollama."""


class GeminiServiceError(Exception):
    """Erreur générique lors de la communication avec l'API Gemini."""

    def __init__(self, message: str, *, error_code: int | None = None, error_message: str | None = None) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.error_message = error_message


class GeminiUnavailableError(GeminiServiceError):
    """Gemini est injoignable (connexion, timeout, clé API absente)."""


class GeminiQuotaExceededError(GeminiServiceError):
    """Le quota / la limite de débit de l'API Gemini a été dépassé."""


class GeminiInvalidResponseError(GeminiServiceError):
    """La réponse de Gemini n'a pas pu être interprétée comme un JSON structuré valide."""


class YoutubeServiceError(Exception):
    """Erreur générique lors de la communication avec YouTube Data API v3."""


class YoutubeQuotaExceeded(YoutubeServiceError):
    """Quota journalier de la clé Data API dépassé (403 quotaExceeded / dailyLimitExceeded)."""


class YoutubeUnavailable(YoutubeServiceError):
    """YouTube Data API est injoignable (timeout, erreur réseau, 5xx)."""


class YoutubeConfigError(YoutubeServiceError):
    """Clé API absente ou invalide (403 keyInvalid, 400) — inutile de réessayer."""


class TTSError(Exception):
    """Erreur générique de synthèse vocale."""


class TTSUnavailableError(TTSError):
    """Le moteur TTS est injoignable ou renvoie une erreur serveur après retries."""


class TTSInvalidVoiceError(TTSError):
    """Le moteur TTS a rejeté la requête (voix inconnue, texte invalide) : inutile de réessayer."""


class AudioAssemblyError(Exception):
    """L'assemblage audio (ffmpeg) a échoué."""


class MediaWebError(Exception):
    """Erreur générique lors de la résolution d'une image web (Commons, Openverse)."""


class MediaFetchRejected(MediaWebError):
    """Téléchargement refusé avant toute lecture réseau : hôte hors liste blanche, IP privée, redirection interdite."""


class MediaFetchTooLarge(MediaWebError):
    """Le corps de la réponse dépasse `MEDIA_MAX_BYTES`, coupé en streaming avant lecture complète."""
