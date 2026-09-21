"""Question de rappel en fin de segment : script, sérialisation, silence, limitation de /more-sections."""

import math
import struct
import wave
from pathlib import Path

import pytest

from app.api.schemas import CourseGenerationResponse
from app.schemas.podcast import PodcastSegment
from app.services.podcast.audio_assembler import TimedTurn, build_timeline
from app.services.podcast.course_serializer import SourceSection, serialize_course
from app.services.podcast.script_generator import _segments_prompt, enforce_budget


def _recall_segment(body_words: int = 300) -> PodcastSegment:
    return PodcastSegment.model_validate({
        "section_ref": 1,
        "title": "Notion",
        "turns": [
            {"speaker": "HOST", "text": "Présentez la notion."},
            {"speaker": "EXPERT", "text": "mot " * body_words},
            {"speaker": "HOST", "text": "Pourquoi le rendement baisse-t-il ?", "think_pause": True},
            {"speaker": "EXPERT", "text": "Parce que les pertes augmentent."},
        ],
    })


def test_enforce_budget_never_truncates_final_recall_pair():
    out = enforce_budget(_recall_segment(), budget=50)

    assert [t.speaker for t in out.turns][-2:] == ["HOST", "EXPERT"]
    assert out.turns[-2].think_pause is True
    assert out.turns[-1].text == "Parce que les pertes augmentent."
    assert len(out.turns) == 3  # HOST d'ouverture + question + réponse ; le long corps est écarté


def test_enforce_budget_without_recall_keeps_previous_behavior():
    seg = PodcastSegment.model_validate({
        "section_ref": 1, "title": "t",
        "turns": [{"speaker": "HOST", "text": "Une phrase."}, {"speaker": "EXPERT", "text": "Une phrase."}],
    })

    assert [t.speaker for t in enforce_budget(seg, budget=50).turns] == ["HOST", "EXPERT"]


def test_recall_question_not_at_segment_end_is_treated_as_regular_turn():
    seg = PodcastSegment.model_validate({
        "section_ref": 1, "title": "t",
        "turns": [
            {"speaker": "HOST", "text": "Question ?", "think_pause": True},
            {"speaker": "EXPERT", "text": "mot " * 300},
            {"speaker": "HOST", "text": "Suite."},
            {"speaker": "EXPERT", "text": "Fin."},
        ],
    })

    assert len(enforce_budget(seg, budget=50).turns) < 4  # tronqué normalement


def test_segments_prompt_asks_for_final_recall_question():
    section = SourceSection(1, "Notion", "mot " * 100, "development")

    prompt = _segments_prompt([section], {1: 100}, "Cours", "conversational")

    assert "question de rappel" in prompt and "think_pause" in prompt


def _course(section: dict) -> CourseGenerationResponse:
    return CourseGenerationResponse.model_validate({
        "mode": "question_only",
        "format": "full_course",
        "meta": {"title": "Cours", "subject": "S", "language": "fr", "generated_at": "2026-01-01T00:00:00Z"},
        "sources": [],
        "introduction": {"quoi": "Intro"},
        "sections": [section],
        "summary": "Résumé",
    })


_BASE_SECTION = {
    "id": "0", "title": "ACV", "quoi": "def", "pourquoi": "raison", "comment": "méca",
    "worked_example": {"statement": "", "steps": [], "result": ""},
}


def test_serialize_passes_challenge_and_check_questions_with_answers():
    section = {
        **_BASE_SECTION,
        "challenge": "Que se passe-t-il si on double les spires ?",
        "check_questions": [
            {"question": "Quelle grandeur change ?", "options": ["Tension", "Fréquence"], "correct_option_indices": [0]},
            {"question": "Q2 ?", "options": ["a", "b"], "correct_option_indices": [1]},
            {"question": "Q3 ignorée ?", "options": ["a", "b"], "correct_option_indices": [0]},
        ],
    }

    text = next(s.text for s in serialize_course(_course(section)) if s.kind == "development")

    assert "Défi : Que se passe-t-il si on double les spires ?" in text
    assert "Questions de rappel : Quelle grandeur change ? Réponse : Tension | Q2 ? Réponse : b" in text
    assert "Q3" not in text  # au plus 2 questions par segment audio


def test_serialize_old_sections_without_cycle_still_work():
    text = next(s.text for s in serialize_course(_course(_BASE_SECTION)) if s.kind == "development")

    assert "Défi" not in text and "Questions de rappel" not in text


def _wav(path: Path, seconds: float, rate: int = 22050) -> Path:
    frames = b"".join(
        struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / rate))) for i in range(int(seconds * rate))
    )
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(frames)
    return path


def test_think_pause_silence_duration_is_exact_in_timeline(tmp_path):
    turns = [
        TimedTurn("HOST", "Question de rappel ?", [_wav(tmp_path / "a.wav", 1.0)], chapter="Notion", think_pause=True),
        TimedTurn("EXPERT", "Réponse.", [_wav(tmp_path / "b.wav", 2.0)]),
    ]

    timeline = build_timeline(turns, pause_turn=0.5, pause_think=4.0)

    assert [v for k, v in timeline.items if k == "silence"] == [4.0, 0.5]
    assert timeline.cues[1].start == pytest.approx(1.0 + 4.0 + 0.5)
    assert timeline.duration == pytest.approx(1.0 + 4.0 + 0.5 + 2.0)
