"""Calcul du prochain reset de quota Gemini — minuit heure du Pacifique, quel que soit le
fuseau horaire système du conteneur (même logique que `_QuotaBreaker` dans
`app/services/course_videos.py`, extraite ici pour être réutilisée par le client Gemini et
l'état de quota partagé, sans coupler les deux domaines)."""

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

_PACIFIC = ZoneInfo("America/Los_Angeles")


def next_pacific_midnight_utc(now: datetime | None = None) -> datetime:
    """Prochain minuit heure du Pacifique, en UTC (aware).

    Paramètres:
        now: horloge injectable pour les tests ; `datetime.now(timezone.utc)` par défaut.
    """
    reference = (now or datetime.now(timezone.utc)).astimezone(_PACIFIC)
    next_midnight = (reference + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return next_midnight.astimezone(timezone.utc)
