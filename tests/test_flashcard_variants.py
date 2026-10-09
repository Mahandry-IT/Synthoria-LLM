import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.core.exceptions import GeminiQuotaExceededError
from app.services import flashcard_variants as fv

SID = uuid.uuid4()
CARDS = [
    {"card_id": "0-0", "front": "Q0 ?", "back": "A"},
    {"card_id": "0-1", "front": "Q1 ?", "back": "B"},
    {"card_id": "1-0", "front": "Q2 ?", "back": "C"},
]


def _review(variant_no: int) -> SimpleNamespace:
    return SimpleNamespace(variant_no=variant_no)


def _raw_variant(text: str, correct: list[int] | None = None) -> dict:
    return {
        "question": text, "choices": ["A", "B"], "explanation": "x",
        "correct_indices": correct if correct is not None else [0],
    }


def test_cards_needing_variants_are_known_cards_waiting_for_a_missing_variant():
    reviews = {(SID, "0-0"): _review(1), (SID, "0-1"): _review(1), (SID, "1-0"): _review(0)}
    variants = {(SID, "0-1"): {1: SimpleNamespace(front="V")}}

    needing = fv.cards_needing_variants(SID, CARDS, reviews, variants, limit=10)

    assert [c["card_id"] for c in needing] == ["0-0"]


def test_cards_needing_variants_is_bounded():
    reviews = {(SID, c["card_id"]): _review(1) for c in CARDS}
    assert len(fv.cards_needing_variants(SID, CARDS, reviews, {}, limit=2)) == 2


def test_parse_variants_numbers_after_existing_and_drops_invalid_or_unrequested():
    variants = {(SID, "0-0"): {1: SimpleNamespace(front="V1"), 2: SimpleNamespace(front="V2")}}
    structured = {
        "cards": [
            {"card_id": "0-0", "variants": [_raw_variant("a"), _raw_variant("b", []), _raw_variant("c")]},
            {"card_id": "9-9", "variants": [_raw_variant("intrus")]},
            {"card_id": "0-1", "variants": [_raw_variant("e", [5])]},
        ]
    }

    rows = fv.parse_variants(structured, SID, {"0-0", "0-1"}, variants)

    # 2 variantes max lues par carte ; la seconde (sans bonne réponse) est écartée.
    assert [(r.card_id, r.variant_no, r.front) for r in rows] == [("0-0", 3, "a")]


def test_prompt_lists_cards_and_previous_wordings():
    variants = {(SID, "0-0"): {1: SimpleNamespace(front="Ancienne variante")}}

    prompt = fv.build_prompt({"meta": {"title": "T"}}, CARDS[:1], variants, SID, Settings(gemini_api_key="k"))

    assert "card_id: 0-0" in prompt and "- Q0 ?" in prompt and "- Ancienne variante" in prompt
    assert "exactement 2 variantes" in prompt


@pytest.fixture
def env(monkeypatch):
    added: list = []

    @asynccontextmanager
    async def factory():
        yield SimpleNamespace(commit=AsyncMock(), add_all=added.extend)

    question = {"question": "Q0 ?", "options": ["A", "B"], "correct_option_indices": [0], "explanation": "e"}
    course = {"sections": [{"id": "0", "check_questions": [question]}]}
    monkeypatch.setattr(
        fv.course_session_repository, "get_by_id", AsyncMock(return_value=SimpleNamespace(gemini_response=course))
    )
    monkeypatch.setattr(
        fv.flashcard_review_repository, "get_for_sessions", AsyncMock(return_value={(SID, "0-0"): _review(1)})
    )
    monkeypatch.setattr(fv.flashcard_variant_repository, "get_for_sessions", AsyncMock(return_value={}))
    return SimpleNamespace(factory=factory, added=added)


@pytest.mark.asyncio
async def test_generate_variants_makes_one_gemini_call_and_stores_two_variants(env):
    gemini = AsyncMock()
    gemini.format_structured.return_value = {
        "cards": [{"card_id": "0-0", "variants": [_raw_variant("v1"), _raw_variant("v2")]}]
    }
    in_flight = {SID}

    await fv.generate_variants(SID, env.factory, gemini, Settings(gemini_api_key="k"), in_flight)

    gemini.format_structured.assert_awaited_once()
    assert [(v.variant_no, v.front) for v in env.added] == [(1, "v1"), (2, "v2")]
    assert in_flight == set()


@pytest.mark.asyncio
async def test_generate_variants_failure_is_swallowed(env):
    gemini = AsyncMock()
    gemini.format_structured.side_effect = GeminiQuotaExceededError("quota")
    in_flight = {SID}

    await fv.generate_variants(SID, env.factory, gemini, Settings(gemini_api_key="k"), in_flight)

    assert env.added == [] and in_flight == set()


@pytest.mark.asyncio
async def test_generate_variants_skips_gemini_when_nothing_is_missing(env, monkeypatch):
    monkeypatch.setattr(
        fv.flashcard_review_repository, "get_for_sessions", AsyncMock(return_value={(SID, "0-0"): _review(0)})
    )
    gemini = AsyncMock()

    await fv.generate_variants(SID, env.factory, gemini, Settings(gemini_api_key="k"), set())

    gemini.format_structured.assert_not_called()
