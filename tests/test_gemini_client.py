from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.core.config import Settings
from app.core.exceptions import (
    GeminiDailyQuotaExceededError,
    GeminiInvalidResponseError,
    GeminiQuotaExceededError,
    GeminiUnavailableError,
)
from app.services.gemini_client import GeminiClient, _classify_quota_error


def _settings(**overrides) -> Settings:
    return Settings(gemini_api_key="fake-key", gemini_max_retries=2, gemini_timeout_seconds=1.0, **overrides)


def _fake_genai_client(generate_content_side_effect=None, generate_content_return_value=None) -> MagicMock:
    fake = MagicMock()
    if generate_content_side_effect is not None:
        fake.models.generate_content.side_effect = generate_content_side_effect
    else:
        fake.models.generate_content.return_value = generate_content_return_value
    return fake


def _quota_exceeded_error() -> Exception:
    """Reproduit la forme structurée (`exc.response.json()`) exposée par le SDK `google-genai` sur
    un vrai 429 — `_parse_gemini_error` n'y lit `error.code`/`error.status` que sous cette forme ;
    un simple `Exception("429 ...")` (comme utilisé ailleurs pour juste déclencher le retry) ne
    suffit pas à faire lever `GeminiQuotaExceededError` en bout de course, seulement à faire
    identifier l'appel comme rate-limited pour le backoff."""
    exc = Exception("429 RESOURCE_EXHAUSTED")
    exc.response = SimpleNamespace(
        json=lambda: {
            "error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "quota exceeded"}
        }
    )
    return exc


def _quota_error_with_violation(quota_id: str) -> Exception:
    """429 avec un détail `QuotaFailure` (forme supposée, non confirmée sur un vrai corps Google —
    voir `_classify_quota_error`), pour tester la classification jour/minute."""
    exc = Exception("429 RESOURCE_EXHAUSTED")
    exc.response = SimpleNamespace(
        json=lambda: {
            "error": {
                "code": 429, "status": "RESOURCE_EXHAUSTED", "message": "quota exceeded",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [{"quotaId": quota_id}],
                    }
                ],
            }
        }
    )
    return exc


def _not_found_error() -> Exception:
    exc = Exception("404 Not Found")
    exc.response = SimpleNamespace(
        json=lambda: {"error": {"code": 404, "status": "NOT_FOUND", "message": "model not found"}}
    )
    return exc


@pytest.mark.asyncio
async def test_search_grounded_returns_text_and_web_sources():
    response = SimpleNamespace(
        text="réponse groundée",
        candidates=[
            SimpleNamespace(
                grounding_metadata=SimpleNamespace(
                    grounding_chunks=[
                        SimpleNamespace(web=SimpleNamespace(title="Source A", uri="https://a.example"))
                    ]
                )
            )
        ],
    )
    fake_client = _fake_genai_client(generate_content_return_value=response)
    client = GeminiClient(_settings(), client=fake_client)

    text, sources = await client.search_grounded("question", "system instruction")

    assert text == "réponse groundée"
    assert sources == [{"type": "web", "label": "Source A", "reference": "https://a.example"}]


@pytest.mark.asyncio
async def test_search_grounded_without_grounding_metadata_returns_empty_sources():
    response = SimpleNamespace(text="ok", candidates=[])
    fake_client = _fake_genai_client(generate_content_return_value=response)
    client = GeminiClient(_settings(), client=fake_client)

    text, sources = await client.search_grounded("question", "system")

    assert text == "ok"
    assert sources == []


@pytest.mark.asyncio
async def test_format_structured_parses_json():
    response = SimpleNamespace(text='{"summary": "ok"}')
    fake_client = _fake_genai_client(generate_content_return_value=response)
    client = GeminiClient(_settings(), client=fake_client)

    result = await client.format_structured("raw answer", response_schema={}, system_instruction="system")

    assert result == {"summary": "ok"}


@pytest.mark.asyncio
async def test_format_structured_invalid_json_raises():
    response = SimpleNamespace(text="not json")
    fake_client = _fake_genai_client(generate_content_return_value=response)
    client = GeminiClient(_settings(), client=fake_client)

    with pytest.raises(GeminiInvalidResponseError):
        await client.format_structured("raw answer", response_schema={}, system_instruction="system")


@pytest.mark.asyncio
async def test_format_structured_falls_back_to_flash_when_lite_quota_exceeded():
    """Le lite et flash ont des quotas Gemini séparés : un lite épuisé (429) ne doit pas faire
    échouer l'appel si flash a encore du quota (bug reproduit : la régénération de section
    échouait alors qu'une nouvelle génération de cours passait, simplement parce qu'elle
    retentait plus de fois et retombait parfois sur un flash-lite pas encore épuisé)."""
    settings = _settings(gemini_model_flash="flash-full", gemini_model_flash_lite="flash-lite")
    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = [
        _quota_exceeded_error(),
        _quota_exceeded_error(),
        SimpleNamespace(text='{"ok": true}'),
    ]
    client = GeminiClient(settings, client=fake_client)

    result = await client.format_structured("raw", response_schema={}, system_instruction="system")

    assert result == {"ok": True}
    calls = fake_client.models.generate_content.call_args_list
    assert calls[0].kwargs["model"] == "flash-lite"  # gemini_max_retries=2 tentatives sur le lite
    assert calls[-1].kwargs["model"] == "flash-full"


@pytest.mark.asyncio
async def test_format_structured_raises_quota_exceeded_when_both_models_exhausted():
    fake_client = _fake_genai_client(generate_content_side_effect=_quota_exceeded_error())
    client = GeminiClient(_settings(), client=fake_client)

    with pytest.raises(GeminiQuotaExceededError):
        await client.format_structured("raw", response_schema={}, system_instruction="system")


@pytest.mark.asyncio
async def test_format_structured_falls_back_to_flash_on_400_schema_error():
    settings = _settings(gemini_model_flash="flash-full", gemini_model_flash_lite="flash-lite")
    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = [
        Exception("400 Bad Request: invalid schema"),
        Exception("400 Bad Request: invalid schema"),
        SimpleNamespace(text='{"ok": true}'),
    ]
    client = GeminiClient(settings, client=fake_client)

    result = await client.format_structured("raw", response_schema={}, system_instruction="system")

    assert result == {"ok": True}
    assert fake_client.models.generate_content.call_args_list[-1].kwargs["model"] == "flash-full"


@pytest.mark.asyncio
async def test_describe_image_uses_flash_lite_model():
    """L'extraction d'images (PDF) doit utiliser le modèle lite, moins coûteux que flash."""
    response = SimpleNamespace(text="Une image décrivant un graphique.")
    fake_client = _fake_genai_client(generate_content_return_value=response)
    settings = _settings(gemini_model_flash="flash-full", gemini_model_flash_lite="flash-lite")
    client = GeminiClient(settings, client=fake_client)

    text = await client.describe_image(b"fake-bytes", "image/png", "system instruction")

    assert text == "Une image décrivant un graphique."
    fake_client.models.generate_content.assert_called_once()
    assert fake_client.models.generate_content.call_args.kwargs["model"] == "flash-lite"


@pytest.mark.asyncio
async def test_describe_image_goes_through_rate_limiter_and_retries():
    """Un 429 sur describe_image doit retenter comme n'importe quel autre appel Gemini."""
    response = SimpleNamespace(text="ok")
    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = [Exception("429 RESOURCE_EXHAUSTED"), response]
    client = GeminiClient(_settings(), client=fake_client)

    text = await client.describe_image(b"fake-bytes", "image/png", "system")

    assert text == "ok"
    assert fake_client.models.generate_content.call_count == 2


@pytest.mark.asyncio
async def test_describe_image_raises_unavailable_without_api_key():
    client = GeminiClient(Settings(gemini_api_key=None))

    with pytest.raises(GeminiUnavailableError):
        await client.describe_image(b"fake-bytes", "image/png", "system")


@pytest.mark.asyncio
async def test_rank_images_sends_one_part_per_image_plus_prompt_uses_flash_lite():
    response = SimpleNamespace(text='{"best_index": 1, "score": 80, "reason": "ok"}')
    fake_client = _fake_genai_client(generate_content_return_value=response)
    settings = _settings(gemini_model_flash="flash-full", gemini_model_flash_lite="flash-lite")
    client = GeminiClient(settings, client=fake_client)

    result = await client.rank_images(
        [(b"img1", "image/webp"), (b"img2", "image/webp")],
        "prompt texte",
        system_instruction="system",
        response_schema={"type": "object"},
    )

    assert result == {"best_index": 1, "score": 80, "reason": "ok"}
    fake_client.models.generate_content.assert_called_once()
    call = fake_client.models.generate_content.call_args
    assert call.kwargs["model"] == "flash-lite"
    contents = call.kwargs["contents"]
    assert len(contents) == 3  # 2 images + le texte du prompt
    assert contents[-1] == "prompt texte"


@pytest.mark.asyncio
async def test_rank_images_raises_invalid_response_on_non_json():
    response = SimpleNamespace(text="not json")
    fake_client = _fake_genai_client(generate_content_return_value=response)
    client = GeminiClient(_settings(), client=fake_client)

    with pytest.raises(GeminiInvalidResponseError):
        await client.rank_images([(b"img", "image/webp")], "prompt", system_instruction="s", response_schema={})


@pytest.mark.asyncio
async def test_rank_images_raises_unavailable_without_api_key():
    client = GeminiClient(Settings(gemini_api_key=None))

    with pytest.raises(GeminiUnavailableError):
        await client.rank_images([(b"img", "image/webp")], "prompt", system_instruction="s", response_schema={})


def test_is_configured_reflects_api_key_presence():
    assert GeminiClient(_settings()).is_configured is True
    assert GeminiClient(Settings(gemini_api_key=None)).is_configured is False


@pytest.mark.asyncio
async def test_rate_limit_retries_then_raises_unavailable():
    """Rate-limit persistant : retry sur flash (gemini_max_retries), puis
    cascade sur flash-lite qui échoue aussi (gemini_max_retries) -> Unavailable."""
    fake_client = _fake_genai_client(generate_content_side_effect=Exception("429 RESOURCE_EXHAUSTED"))
    client = GeminiClient(_settings(), client=fake_client)

    with pytest.raises(GeminiUnavailableError):
        await client.search_grounded("question", "system")

    assert fake_client.models.generate_content.call_count == 4  # 2 modèles x gemini_max_retries=2


@pytest.mark.asyncio
async def test_transient_error_retries_then_raises_unavailable():
    fake_client = _fake_genai_client(generate_content_side_effect=Exception("connexion impossible"))
    client = GeminiClient(_settings(), client=fake_client)

    with pytest.raises(GeminiUnavailableError):
        await client.search_grounded("question", "system")

    assert fake_client.models.generate_content.call_count == 4  # 2 modèles x gemini_max_retries=2


@pytest.mark.asyncio
async def test_missing_api_key_raises_unavailable():
    client = GeminiClient(Settings(gemini_api_key=None))

    with pytest.raises(GeminiUnavailableError):
        await client.search_grounded("question", "system")


# --- Cascade flash -> flash-lite ---


@pytest.mark.asyncio
async def test_cascade_falls_back_to_lite_on_quota_exceeded():
    """Flash épuise son quota (429) -> bascule sur flash-lite qui répond."""
    response = SimpleNamespace(text="réponse flash-lite", candidates=[])
    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = [
        Exception("429 RESOURCE_EXHAUSTED"),
        Exception("429 RESOURCE_EXHAUSTED"),
        response,
    ]
    client = GeminiClient(_settings(), client=fake_client)

    text, _ = await client.search_grounded("question", "system")

    assert text == "réponse flash-lite"
    calls = fake_client.models.generate_content.call_args_list
    assert calls[0].kwargs["model"] == client._settings.gemini_model_flash
    assert calls[-1].kwargs["model"] == client._settings.gemini_model_flash_lite


@pytest.mark.asyncio
async def test_cascade_falls_back_to_lite_on_unavailable():
    """Flash injoignable (erreur non rate-limit) -> bascule sur flash-lite qui répond."""
    response = SimpleNamespace(text="réponse flash-lite", candidates=[])
    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = [
        Exception("connexion impossible"),
        Exception("connexion impossible"),
        response,
    ]
    client = GeminiClient(_settings(), client=fake_client)

    text, _ = await client.search_grounded("question", "system")

    assert text == "réponse flash-lite"


@pytest.mark.asyncio
async def test_cascade_propagates_error_when_all_models_fail():
    fake_client = _fake_genai_client(generate_content_side_effect=Exception("429 RESOURCE_EXHAUSTED"))
    client = GeminiClient(_settings(), client=fake_client)

    with pytest.raises(GeminiUnavailableError):
        await client.search_grounded("question", "system")


@pytest.mark.asyncio
async def test_cascade_does_not_fall_back_on_first_model_success():
    response = SimpleNamespace(text="ok flash", candidates=[])
    fake_client = _fake_genai_client(generate_content_return_value=response)
    client = GeminiClient(_settings(), client=fake_client)

    text, _ = await client.search_grounded("question", "system")

    assert text == "ok flash"
    assert fake_client.models.generate_content.call_count == 1


# ─── Limite de débit (RPM) ────────────────────────────────────


@pytest.mark.asyncio
async def test_every_gemini_call_goes_through_the_rate_limiter():
    """Chaque appel réseau (initial et retries) réserve un créneau avant de partir."""
    response = SimpleNamespace(text='{"ok": true}', candidates=[])
    fake_client = _fake_genai_client(generate_content_return_value=response)
    client = GeminiClient(_settings(), client=fake_client)
    acquired = 0
    original = client._rate_limiter.acquire

    async def counting_acquire():
        nonlocal acquired
        acquired += 1
        await original()

    client._rate_limiter.acquire = counting_acquire

    await client.format_structured("raw", response_schema={}, system_instruction="system")

    assert acquired == 1  # un seul appel réseau ici : un seul créneau réservé


@pytest.mark.asyncio
async def test_retries_each_reserve_their_own_slot():
    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = [Exception("connexion impossible"), SimpleNamespace(text="ok", candidates=[])]
    client = GeminiClient(_settings(), client=fake_client)  # gemini_max_retries=2
    acquired = 0
    original = client._rate_limiter.acquire

    async def counting_acquire():
        nonlocal acquired
        acquired += 1
        await original()

    client._rate_limiter.acquire = counting_acquire

    await client.search_grounded("question", "system")

    assert acquired == 2  # premier essai échoué + retry réussi : deux créneaux


# ─── Instrumentation (track_calls) ─────────────────────────────


@pytest.mark.asyncio
async def test_track_calls_logs_summary_with_counts_by_model_and_method(caplog):
    settings = _settings(gemini_model_flash="flash-full", gemini_model_flash_lite="flash-lite")
    fake_client = _fake_genai_client(generate_content_return_value=SimpleNamespace(text='{"ok": true}'))
    client = GeminiClient(settings, client=fake_client)

    with caplog.at_level("INFO", logger="app.services.gemini_client"):
        with client.track_calls():
            await client.format_structured("raw", response_schema={}, system_instruction="system")
            await client.describe_image(b"img", "image/png", "system")

    summaries = [r for r in caplog.records if r.message == "gemini_calls_summary"]
    assert len(summaries) == 1
    assert summaries[0].total_calls == 2
    assert summaries[0].by_method == {"format_structured": 1, "describe_image": 1}
    assert summaries[0].by_model == {"flash-lite": 2}


@pytest.mark.asyncio
async def test_calls_outside_track_calls_context_do_not_raise():
    fake_client = _fake_genai_client(generate_content_return_value=SimpleNamespace(text='{"ok": true}'))
    client = GeminiClient(_settings(), client=fake_client)

    result = await client.format_structured("raw", response_schema={}, system_instruction="system")

    assert result == {"ok": True}


@pytest.mark.asyncio
async def test_track_calls_does_not_leak_entries_across_separate_calls():
    fake_client = _fake_genai_client(generate_content_return_value=SimpleNamespace(text='{"ok": true}'))
    client = GeminiClient(_settings(), client=fake_client)

    with client.track_calls():
        await client.format_structured("raw", response_schema={}, system_instruction="system")

    # Hors contexte : aucun appel ne doit être comptabilisé (pas d'état résiduel entre requêtes).
    from app.services.gemini_client import _call_log

    assert _call_log.get() is None


# ─── Classification 429 jour/minute + fail-fast (lot 2) ────────


def test_classify_quota_error_detects_per_day_violation():
    assert _classify_quota_error(_quota_error_with_violation("GenerateContentPerDayPerProjectPerModel")) == "day"


def test_classify_quota_error_detects_per_minute_violation():
    assert _classify_quota_error(_quota_error_with_violation("GenerateContentPerMinutePerProjectPerModel")) == "minute"


def test_classify_quota_error_returns_unknown_without_details():
    assert _classify_quota_error(_quota_exceeded_error()) == "unknown"


def test_classify_quota_error_returns_unknown_for_non_gemini_exception():
    assert _classify_quota_error(Exception("boom")) == "unknown"


@pytest.mark.asyncio
async def test_call_with_retry_raises_daily_quota_immediately_without_retry():
    """Isolé au niveau de `_call_with_retry` (pas `format_structured`, qui a son propre repli
    lite→flash au-dessus : le fail-fast s'applique par modèle, `func` n'est donc appelé qu'une
    fois PAR MODÈLE essayé, jamais `gemini_max_retries` fois pour le même modèle)."""
    fake_client = _fake_genai_client(generate_content_side_effect=_quota_error_with_violation("PerDayPerProject"))
    client = GeminiClient(_settings(), client=fake_client)  # gemini_max_retries=2

    with pytest.raises(GeminiDailyQuotaExceededError) as exc_info:
        await client._call_with_retry(
            lambda model: fake_client.models.generate_content(model=model), "flash-lite"
        )

    assert fake_client.models.generate_content.call_count == 1  # aucun retry gaspillé
    assert exc_info.value.retry_at is not None


@pytest.mark.asyncio
async def test_call_with_retry_raises_unavailable_immediately_on_404_without_retry():
    fake_client = _fake_genai_client(generate_content_side_effect=_not_found_error())
    client = GeminiClient(_settings(), client=fake_client)

    with pytest.raises(GeminiUnavailableError):
        await client._call_with_retry(
            lambda model: fake_client.models.generate_content(model=model), "flash-lite"
        )

    assert fake_client.models.generate_content.call_count == 1


@pytest.mark.asyncio
async def test_call_with_retry_still_retries_on_unclassified_quota_error():
    """Régression : un 429 sans détail `QuotaFailure` exploitable (classification "unknown")
    doit continuer à être retenté comme avant le lot 2 — jamais de fail-fast incertain."""
    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = [_quota_exceeded_error(), SimpleNamespace(text='{"ok": true}')]
    client = GeminiClient(_settings(), client=fake_client)

    result = await client.format_structured("raw", response_schema={}, system_instruction="system")

    assert result == {"ok": True}
    assert fake_client.models.generate_content.call_count == 2