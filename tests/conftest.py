import pytest


@pytest.fixture(autouse=True)
def _no_youtube_lookup(monkeypatch):
    """Pas d'appel réseau YouTube (ni recherche Gemini supplémentaire) dans les tests existants."""
    monkeypatch.setenv("COURSE_VIDEOS_ENABLED", "false")
