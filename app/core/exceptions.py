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
