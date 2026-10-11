from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException

from app.api.routes import _DEFAULT_QUOTA_RETRY_AFTER_SECONDS, _gemini_http_errors
from app.core.exceptions import (
    GeminiDailyQuotaExceededError,
    GeminiInvalidResponseError,
    GeminiQuotaExceededError,
    GeminiUnavailableError,
)


def test_daily_quota_exceeded_sets_retry_after_from_retry_at():
    retry_at = datetime.now(timezone.utc) + timedelta(hours=3)

    with pytest.raises(HTTPException) as exc_info:
        with _gemini_http_errors():
            raise GeminiDailyQuotaExceededError("quota jour dépassé", retry_at=retry_at)

    exc = exc_info.value
    assert exc.status_code == 429
    # Tolérance large : le temps écoulé entre le raise et l'assertion est négligeable.
    assert 10700 <= int(exc.headers["Retry-After"]) <= 10800


def test_daily_quota_exceeded_falls_back_to_default_without_retry_at():
    with pytest.raises(HTTPException) as exc_info:
        with _gemini_http_errors():
            raise GeminiDailyQuotaExceededError("quota jour dépassé", retry_at=None)

    exc = exc_info.value
    assert exc.status_code == 429
    assert exc.headers["Retry-After"] == str(_DEFAULT_QUOTA_RETRY_AFTER_SECONDS)


def test_daily_quota_exceeded_retry_after_never_negative_for_a_past_retry_at():
    retry_at = datetime.now(timezone.utc) - timedelta(minutes=5)  # déjà passé

    with pytest.raises(HTTPException) as exc_info:
        with _gemini_http_errors():
            raise GeminiDailyQuotaExceededError("quota jour dépassé", retry_at=retry_at)

    assert int(exc_info.value.headers["Retry-After"]) >= 1


def test_plain_quota_exceeded_uses_default_retry_after():
    with pytest.raises(HTTPException) as exc_info:
        with _gemini_http_errors():
            raise GeminiQuotaExceededError("rate limit minute")

    exc = exc_info.value
    assert exc.status_code == 429
    assert exc.headers["Retry-After"] == str(_DEFAULT_QUOTA_RETRY_AFTER_SECONDS)


def test_unavailable_error_has_no_retry_after_header():
    with pytest.raises(HTTPException) as exc_info:
        with _gemini_http_errors():
            raise GeminiUnavailableError("indisponible")

    exc = exc_info.value
    assert exc.status_code == 503
    assert not exc.headers or "Retry-After" not in exc.headers


def test_invalid_response_maps_to_502():
    with pytest.raises(HTTPException) as exc_info:
        with _gemini_http_errors():
            raise GeminiInvalidResponseError("json invalide")

    assert exc_info.value.status_code == 502


def test_plain_quota_exceeded_uses_google_retry_delay_rounded_up():
    with pytest.raises(HTTPException) as exc_info:
        with _gemini_http_errors():
            raise GeminiQuotaExceededError("rate limit minute", retry_after_seconds=37.2)

    assert exc_info.value.headers["Retry-After"] == "38"
