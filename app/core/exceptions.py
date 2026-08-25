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


# --- HuggingFace Video Generation ---


class HFVideoServiceError(Exception):
    """Erreur générique lors de la communication avec l'API HF Inference."""

    def __init__(self, message: str, *, error_code: int | None = None) -> None:
        super().__init__(message)
        self.error_code = error_code


class HFVideoUnavailableError(HFVideoServiceError):
    """HF Inference injoignable (connexion, timeout, token absent)."""


class HFVideoRateLimitError(HFVideoServiceError):
    """Quota / rate limit HF dépassé (429)."""


class HFVideoGenerationError(HFVideoServiceError):
    """La génération vidéo a échoué côté HF (500/502/503)."""