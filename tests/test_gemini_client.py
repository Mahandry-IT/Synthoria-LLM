from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.core.config import Settings
from app.core.exceptions import (
    GeminiInvalidResponseError,
    GeminiQuotaExceededError,
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
async def test_quota_error_raises_immediately():
    fake_client = _fake_genai_client(generate_content_side_effect=Exception("429 quota exceeded"))
    client = GeminiClient(_settings(), client=fake_client)

    with pytest.raises(GeminiQuotaExceededError):
        await client.search_grounded("question", "system")


@pytest.mark.asyncio
async def test_transient_error_retries_then_raises_unavailable():
    fake_client = _fake_genai_client(generate_content_side_effect=Exception("connexion impossible"))
    client = GeminiClient(_settings(), client=fake_client)

    with pytest.raises(GeminiUnavailableError):
        await client.search_grounded("question", "system")

    assert fake_client.models.generate_content.call_count == 2


@pytest.mark.asyncio
async def test_missing_api_key_raises_unavailable():
    client = GeminiClient(Settings(gemini_api_key=None))

    with pytest.raises(GeminiUnavailableError):
        await client.search_grounded("question", "system")