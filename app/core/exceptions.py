class OllamaServiceError(Exception):
    """Erreur générique lors de la communication avec Ollama."""


class OllamaUnavailableError(OllamaServiceError):
    """Ollama est injoignable (connexion, timeout)."""


class OllamaModelNotFoundError(OllamaServiceError):
    """Le modèle demandé n'existe pas sur l'instance Ollama."""
