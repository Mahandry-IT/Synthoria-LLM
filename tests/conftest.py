import os

import pytest

# `app.main.app` est instancié à l'import, donc son RateLimitMiddleware (mémoire, par IP) est
# partagé par toute la session : avec la limite par défaut (30/min), les tests de routes
# s'enchaînent assez vite pour recevoir des 429 parasites. Fixé avant tout import de l'app.
os.environ.setdefault("RATE_LIMIT_PER_MINUTE", "100000")


@pytest.fixture(autouse=True)
def _no_youtube_lookup(monkeypatch):
    """Pas d'appel réseau YouTube (ni recherche Gemini supplémentaire) dans les tests existants.

    `YOUTUBE_API_KEY` est aussi neutralisée : un test qui active `course_videos_enabled=True` pour
    exercer `attach_verified_videos` ne doit jamais dépendre de l'absence/présence d'une vraie clé
    dans le `.env` local du poste qui l'exécute (sinon appel réseau réel à YouTube Data API).
    """
    monkeypatch.setenv("COURSE_VIDEOS_ENABLED", "false")
    monkeypatch.setenv("YOUTUBE_API_KEY", "")
