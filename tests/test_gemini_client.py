from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.core.config import Settings
from app.core.exceptions import (
    GeminiInvalidResponseError,
    GeminiUnavailableError,
)
from app.services.gemini_client import GeminiClient


def _settings(**overrides) -> Settings:
    return Settings(gemini_api_key="fake-key", gemini_max_retries=2, gemini_timeout_seconds=1.0, **overrides)


def _fake_genai_client(generate_content_side_effect=None, generate_content_return_value=None) -> MagicMock:
    fake = MagicMock()
    if generate_content_side_effect is not None:
        fake.models.generate_content.side_effect = generate_content_side_effect
    else:
        fake.models.generate_content.return_value = generate_content_return_value
    return fake


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