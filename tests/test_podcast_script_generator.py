from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.core.exceptions import GeminiUnavailableError
from app.schemas.podcast import PodcastFrame, PodcastScript, PodcastSegment, PodcastSegmentsBatch, PodcastTurn
from app.services.podcast.course_serializer import SourceSection
from app.services.podcast.script_generator import (
    ScriptRequest,
    compute_word_budgets,
    enforce_budget,
    estimate_duration_seconds,
    generate_script,
    script_word_count,
)


def _sections(n_dev: int = 3) -> list[SourceSection]:
    sections = [SourceSection(0, "Introduction", "mot " * 40, "intro")]
    sections += [SourceSection(i + 1, f"Notion {i}", "mot " * (100 * (i + 1)), "development") for i in range(n_dev)]
    n = len(sections)
    sections += [
        SourceSection(n, "Pièges fréquents", "mot " * 50, "pitfalls"),
        SourceSection(n + 1, "Synthèse", "résumé du cours", "summary"),
        SourceSection(n + 2, "Pour aller plus loin", "suite possible", "next_steps"),
    ]
    return sections


def _seg(ref: int, title: str = "t", text: str = "Une phrase.") -> dict:
    return {"section_ref": ref, "title": title, "turns": [{"speaker": "HOST", "text": text}, {"speaker": "EXPERT", "text": text}]}


FRAME = {
    "title": "Titre du podcast",
    "intro_turns": [{"speaker": "HOST", "text": "Bienvenue."}],
    "outro_turns": [{"speaker": "EXPERT", "text": "Au revoir."}],
}


def _settings(batch: int = 4) -> Settings:
    return Settings(gemini_api_key="k", podcast_script_batch_size=batch)


def _request(minutes: int = 10) -> ScriptRequest:
    return ScriptRequest(course_title="Le cours", style="conversational", target_minutes=minutes)


def _fake_gemini(segment_refs_by_call=None, fail_segments: int = 0, frame=FRAME):
    """Renvoie des segments pour les section_ref demandés dans le prompt (indices lus dans <course_data>)."""
    import re

    state = {"seg_calls": 0}

    async def format_structured(raw_answer, system_instruction, *, response_schema=None):
        if response_schema is PodcastFrame:
            return frame
        state["seg_calls"] += 1
        if state["seg_calls"] <= fail_segments:
            return {"segments": "pas une liste"}
        refs = [int(x) for x in re.findall(r'<course_data index="(\d+)"', raw_answer)]
        return {"segments": [_seg(r, f"Seg {r}") for r in refs]}

    client = AsyncMock()
    client.format_structured.side_effect = format_structured
    return client, state


# ─── Calculs déterministes ───────────────────────────────────


def test_word_budgets_are_proportional_with_floor_and_skip_wrap_up_sections():
    budgets = compute_word_budgets(_sections(), target_minutes=10)
    assert set(budgets) == {0, 1, 2, 3, 4}  # ni synthèse ni suite
    assert budgets[3] > budgets[2] > budgets[1]
    assert all(b >= 60 for b in budgets.values())
    assert 10 * 150 * 0.8 <= sum(budgets.values()) <= 10 * 150


def test_word_budgets_empty_without_segment_sections():
    assert compute_word_budgets([SourceSection(0, "Synthèse", "x", "summary")], 10) == {}


def test_enforce_budget_merges_same_speaker_and_truncates_over_1_5x():
    turns = [PodcastTurn(speaker="HOST", text="a b c"), PodcastTurn(speaker="HOST", text="d e")]
    turns += [PodcastTurn(speaker="EXPERT", text=" ".join(["mot"] * 40)) for _ in range(3)]
    seg = PodcastSegment(section_ref=0, title="t", turns=turns)

    out = enforce_budget(seg, budget=50)

    assert out.turns[0].text == "a b c d e"          # fusion
    assert sum(len(t.text.split()) for t in out.turns) <= 75 + 40  # une réplique ne dépasse que la dernière
    assert len(out.turns) < 4                        # tronqué


def test_enforce_budget_always_keeps_one_turn():
    seg = PodcastSegment(section_ref=0, title="t", turns=[PodcastTurn(speaker="HOST", text=" ".join(["m"] * 500))])
    assert len(enforce_budget(seg, budget=10).turns) == 1


def test_duration_estimate_from_word_count():
    script = PodcastScript(
        title="t",
        intro_turns=[PodcastTurn(speaker="HOST", text=" ".join(["m"] * 150))],
        segments=[],
        outro_turns=[],
    )
    assert script_word_count(script) == 150
    assert estimate_duration_seconds(script) == pytest.approx(60.0)


# ─── Génération ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_generate_script_happy_path_orders_segments_and_batches():
    client, state = _fake_gemini()

    script = await generate_script(_sections(3), _request(), client, _settings(batch=2))

    assert [s.section_ref for s in script.segments] == [0, 1, 2, 3, 4]  # intro, 3 dev, pièges
    assert state["seg_calls"] == 3                                        # 5 sections / lots de 2
    assert script.title == "Titre du podcast"
    prompts = [c.kwargs["raw_answer"] for c in client.format_structured.call_args_list]
    assert any("<course_data" in p and "Notion 0" in p for p in prompts)
    # La synthèse et la suite alimentent la conclusion, pas un segment
    frame_prompt = next(p for p, c in zip(prompts, client.format_structured.call_args_list)
                        if c.kwargs["response_schema"] is PodcastFrame)
    assert "résumé du cours" in frame_prompt and "suite possible" in frame_prompt


@pytest.mark.asyncio
async def test_generate_script_retries_once_with_validation_error():
    client, state = _fake_gemini(fail_segments=1)

    script = await generate_script(_sections(1), _request(), client, _settings())

    assert state["seg_calls"] == 2
    retry_prompt = client.format_structured.call_args_list[1].kwargs["raw_answer"]
    assert "invalide" in retry_prompt
    assert [s.title for s in script.segments][:1] == ["Seg 0"]


@pytest.mark.asyncio
async def test_generate_script_falls_back_deterministically_when_output_stays_invalid():
    client, _ = _fake_gemini(fail_segments=99)

    script = await generate_script(_sections(1), _request(), client, _settings())

    assert len(script.segments) == 3                         # aucune section perdue
    dev = next(s for s in script.segments if s.title == "Notion 0")
    assert dev.turns[0].speaker == "HOST" and "Notion 0" in dev.turns[0].text
    assert dev.turns[-1].speaker == "EXPERT"


@pytest.mark.asyncio
async def test_generate_script_regenerates_missing_sections_then_falls_back():
    calls = {"n": 0}

    async def format_structured(raw_answer, system_instruction, *, response_schema=None):
        if response_schema is PodcastFrame:
            return FRAME
        calls["n"] += 1
        if calls["n"] == 1:  # 1er jet : n'écrit que la section 0
            return {"segments": [_seg(0, "Seg 0")]}
        return {"segments": []}  # la régénération échoue aussi

    client = AsyncMock()
    client.format_structured.side_effect = format_structured

    script = await generate_script(_sections(1), _request(), client, _settings())

    assert calls["n"] == 2
    assert [s.section_ref for s in script.segments] == [0, 1, 2]
    assert script.segments[0].title == "Seg 0"
    assert script.segments[1].title == "Notion 0"          # repli déterministe
    regen_prompt = client.format_structured.call_args_list[1].kwargs["raw_answer"]
    assert 'index="0"' not in regen_prompt and 'index="1"' in regen_prompt


@pytest.mark.asyncio
async def test_generate_script_frame_falls_back_when_invalid():
    client, _ = _fake_gemini(frame={"title": ""})

    script = await generate_script(_sections(1), _request(), client, _settings())

    assert script.title == "Le cours"
    assert script.intro_turns and script.outro_turns


@pytest.mark.asyncio
async def test_generate_script_propagates_gemini_unavailable():
    client = AsyncMock()
    client.format_structured.side_effect = GeminiUnavailableError("indisponible")

    with pytest.raises(GeminiUnavailableError):
        await generate_script(_sections(1), _request(), client, _settings())


@pytest.mark.asyncio
async def test_generate_script_caps_segments():
    client, _ = _fake_gemini()
    settings = Settings(gemini_api_key="k", podcast_max_segments=2)

    script = await generate_script(_sections(5), _request(), client, settings)

    assert len(script.segments) == 2


def test_prompt_marks_course_content_as_data():
    from app.services.podcast.script_generator import _instructions

    assert "<course_data>" in _instructions() and "data" in _instructions().lower()


@pytest.mark.asyncio
async def test_segments_batch_schema_is_flat_and_valid():
    batch = PodcastSegmentsBatch.model_validate({"segments": [_seg(0)]})
    assert batch.segments[0].turns[1].speaker == "EXPERT"
