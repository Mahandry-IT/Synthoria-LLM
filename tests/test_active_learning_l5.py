from pathlib import Path

from app.api.routes import _api_pretest
from app.api.schemas import ApiPlannedSection
from app.schemas.course_generation import CoursePlanSchema
from app.schemas.podcast import PodcastTurn
from app.services.course_plan_generator import _format_section
from app.services.podcast.audio_assembler import TimedTurn, build_timeline
from app.services.podcast.script_generator import _merge_same_speaker


def _question(text: str = "Q ?") -> dict:
    return {
        "question": text, "choices": ["A", "B"], "correct_indices": [0], "difficulty": "normale",
        "explanation": "", "requires_calculation": False,
    }


def _plan(pretest: list[dict]) -> CoursePlanSchema:
    def section(type_: str, title: str, order: int) -> dict:
        return {"type": type_, "title": title, "objective": "o", "subtopics": [], "order": order}

    return CoursePlanSchema.model_validate({
        "meta": {"title": "T", "subject": "S", "language": "fr"},
        "planned_sections": [section("introduction", "Intro", 1), section("development", "Principe", 2)],
        "pretest": pretest,
    })


def test_pretest_keeps_one_question_per_existing_development_section():
    plan = _plan([
        {"section_title": "principe", "question": _question("Q1")},
        {"section_title": "Principe", "question": _question("Doublon")},  # 2e question : ignorée
        {"section_title": "Intro", "question": _question("Intro n'est pas un développement")},
        {"section_title": "Inconnue", "question": _question("Section absente")},
    ])

    items = _api_pretest(plan)

    assert [(i.section_title, i.question.question) for i in items] == [("principe", "Q1")]
    assert items[0].question.correct_option_indices == [0]


def test_plan_without_pretest_stays_valid():
    assert _api_pretest(_plan([])) == []


def test_mastery_flag_asks_for_condensed_version():
    known = ApiPlannedSection(type="development", title="Principe", objective="o", subtopics=[], order=2, mastery="known")
    normal = known.model_copy(update={"mastery": None})

    assert "CONDENSÉE" in _format_section(known, detailed=True)
    assert "CONDENSÉE" not in _format_section(normal, detailed=True)
    assert "CONDENSÉE" not in _format_section(known, detailed=False)


def _wav_duration(_: Path) -> float:
    return 2.0


def test_think_pause_adds_dedicated_silence_after_recall_question():
    turns = [
        TimedTurn("HOST", "Question de rappel", [Path("a.wav")], "Segment", think_pause=True),
        TimedTurn("EXPERT", "Réponse", [Path("b.wav")]),
    ]

    timeline = build_timeline(turns, duration_of=_wav_duration, pause_turn=0.5, pause_think=5.0)

    assert [i for i in timeline.items if i[0] == "silence"] == [("silence", 5.0), ("silence", 0.5)]
    assert timeline.cues[1].start == 2.0 + 5.0 + 0.5
    assert timeline.duration == 2.0 + 5.0 + 0.5 + 2.0


def test_no_think_pause_after_last_turn():
    timeline = build_timeline([TimedTurn("HOST", "x", [Path("a.wav")], "S", think_pause=True)], duration_of=_wav_duration)

    assert timeline.duration == 2.0


def test_merging_same_speaker_keeps_think_pause():
    merged = _merge_same_speaker([
        PodcastTurn(speaker="HOST", text="Bien."),
        PodcastTurn(speaker="HOST", text="Pourquoi ?", think_pause=True),
        PodcastTurn(speaker="EXPERT", text="Parce que."),
    ])

    assert [(t.speaker, t.think_pause) for t in merged] == [("HOST", True), ("EXPERT", False)]
