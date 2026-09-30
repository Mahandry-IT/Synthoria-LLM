from datetime import datetime, timezone

from app.services.quota_reset import next_pacific_midnight_utc


def test_returns_midnight_pacific_converted_to_utc():
    # 2026-06-15 12:00 UTC = 2026-06-15 05:00 PDT (UTC-7, heure d'été) -> minuit suivant = 2026-06-16 00:00 PDT
    now = datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc)

    result = next_pacific_midnight_utc(now)

    assert result == datetime(2026, 6, 16, 7, 0, tzinfo=timezone.utc)


def test_always_returns_a_future_moment():
    now = datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)

    result = next_pacific_midnight_utc(now)

    assert result > now


def test_crosses_dst_boundary_correctly():
    """Californie : passage à l'heure d'été le 2e dimanche de mars (2026 -> 8 mars).
    Vérifie que le calcul reste correct de part et d'autre du changement d'offset UTC."""
    before_dst = datetime(2026, 3, 7, 12, 0, tzinfo=timezone.utc)  # encore PST (UTC-8)
    after_dst = datetime(2026, 3, 9, 12, 0, tzinfo=timezone.utc)  # déjà PDT (UTC-7)

    result_before = next_pacific_midnight_utc(before_dst)
    result_after = next_pacific_midnight_utc(after_dst)

    assert result_before == datetime(2026, 3, 8, 8, 0, tzinfo=timezone.utc)  # minuit PST -> +8h UTC
    assert result_after == datetime(2026, 3, 10, 7, 0, tzinfo=timezone.utc)  # minuit PDT -> +7h UTC


def test_defaults_to_current_time_when_now_is_none():
    result = next_pacific_midnight_utc()

    assert result > datetime.now(timezone.utc)
