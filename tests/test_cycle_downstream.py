"""Consommateurs aval d'une section au format cycle : flashcards (Leitner), questions de rappel du podcast,
introduction, assistant et exercices lisent bien une section avec cycle ; les anciennes sections ne changent pas."""

import random
from typing import Any

from app.api.schemas import CourseGenerationResponse
from app.schemas.course_generation import Section
from app.services.course_generator import _map_sections_to_course_sections, is_incomplete_section_dict
from app.services.course_regenerator import _is_introduction, planned_sections_from_course
from app.services.exercise_generator import build_prompt, course_is_ready, usable_sections
from app.services.leitner import flashcards_from_course
from app.services.lesson_context import _section_text
from app.services.podcast.course_serializer import serialize_course
from tests.cycle_fixtures import cycle_section, old_format_section


def _stored(*sections: dict[str, Any], mutate=None) -> list[dict[str, Any]]:
    """Sections au format de session persistée (`model_dump` d'API), ids 0, 1, ..."""
    mapped = _map_sections_to_course_sections([Section.model_validate(s) for s in sections], rng=random.Random(0))
    return [s.model_dump(mode="json") for s in mapped]


def _numeric_question() -> dict[str, Any]:
    return {
        "question": "Combien d'adresses dans un /26 ?", "kind": "numeric", "expected_value": 64.0, "unit": "adresses",
        "difficulty": "normale", "explanation": "2 puissance 6.", "requires_calculation": True,
    }


def _qcm_two_questions(section: dict) -> None:
    second = {**section["cycle"]["trial"]["questions"][0], "question": "Quel masque pour 64 adresses ?",
              "choices": ["/24", "/26", "/28"], "correct_indices": [1], "explanation": "Un /26 donne 64 adresses.",
              "explanation_per_choice": []}
    section["cycle"]["trial"]["questions"] = [section["cycle"]["trial"]["questions"][0], second, _numeric_question()]


def _exercise_card(section: dict) -> None:
    section["cycle"]["trial"] = {
        "kind": "exercise_card",
        "card": {"statement": "Découpe 10.0.0.0/24 en 4.", "hint": "Divise par 4.", "steps": ["256/4 = 64", "Masque /26"], "result": "4 x /26"},
    }


# ─── Flashcards ──────────────────────────────────────────────────────────────────────────


def test_a_cycle_section_gets_cards_from_its_challenge_then_its_qcm_choice_questions():
    (stored,) = _stored(cycle_section("Sous-réseaux", _qcm_two_questions))

    cards = flashcards_from_course({"sections": [stored]})

    assert [c["card_id"] for c in cards] == ["0-0", "0-1", "0-2"]  # défi, puis les deux questions à choix ; pas la numérique
    challenge, first, second = cards
    assert challenge["front"].startswith("On découpe 192.168.1.0/24")
    assert len(challenge["correct_indices"]) == 2  # cases à cocher : plusieurs bonnes réponses
    assert {challenge["choices"][i] for i in challenge["correct_indices"]} == {"Chaque part a 64 adresses", "Les parts couvrent tout le réseau"}
    assert "Chaque part a 64 adresses" in challenge["back"] and "Quatre parts égales" in challenge["back"]
    assert challenge["explanation"] == "Quatre parts égales de 256 adresses font 64 adresses chacune."
    assert first["front"] == "Combien d'adresses par sous-réseau ?" and first["choices"][first["correct_indices"][0]] == "64"
    assert second["choices"][second["correct_indices"][0]] == "/26"
    assert all(c["section_ref"] == "0" for c in cards)


def test_cards_follow_the_shuffled_options_and_stay_consistent():
    (stored,) = _stored(cycle_section("Sous-réseaux"))

    for card in flashcards_from_course({"sections": [stored]}):
        assert card["correct_indices"] and all(0 <= i < len(card["choices"]) for i in card["correct_indices"])
        assert all(card["choices"][i] in card["back"] for i in card["correct_indices"])


def test_an_exercise_card_trial_gives_only_the_challenge_card():
    (stored,) = _stored(cycle_section("Sous-réseaux", _exercise_card))

    cards = flashcards_from_course({"sections": [stored]})

    assert [c["card_id"] for c in cards] == ["0-0"]


def test_a_cycle_section_never_loses_all_its_flashcards():
    (stored,) = _stored(cycle_section())
    assert stored["check_questions"] == []  # c'est bien ce qui les faisait disparaître

    assert len(flashcards_from_course({"sections": [stored]})) >= 2


def test_old_sections_keep_exactly_their_check_question_cards():
    section = {
        "id": "3", "title": "Ancienne",
        "check_questions": [
            {"question": "Q1 ?", "options": ["a", "b", "c"], "correct_option_indices": [2], "explanation": "parce que"},
            {"question": "Sans bonne réponse ?", "options": ["a"], "correct_option_indices": [5], "explanation": ""},
            {"question": "Q3 ?", "options": ["x", "y"], "correct_option_indices": [0, 1], "explanation": ""},
        ],
    }

    cards = flashcards_from_course({"sections": [section]})

    assert cards == [
        {"card_id": "3-0", "front": "Q1 ?", "back": "c\nparce que", "section_ref": "3",
         "choices": ["a", "b", "c"], "correct_indices": [2], "explanation": "parce que"},
        {"card_id": "3-2", "front": "Q3 ?", "back": "x ; y", "section_ref": "3",
         "choices": ["x", "y"], "correct_indices": [0, 1], "explanation": ""},
    ]


def test_mixed_course_keeps_old_card_ids_and_adds_the_cycle_ones():
    old = _stored(old_format_section("Ancienne"))[0]
    old["check_questions"] = [{"question": "Q ?", "options": ["a", "b"], "correct_option_indices": [0], "explanation": ""}]
    cycle = _stored(cycle_section("Cycle"))[0]
    cycle["id"] = "1"

    ids = [c["card_id"] for c in flashcards_from_course({"sections": [old, cycle]})]

    assert ids == [f"{old['id']}-0", "1-0", "1-1"]


def test_broken_stored_cycle_does_not_raise():
    cards = flashcards_from_course({"sections": [{"id": "0", "cycle": {"challenge": None, "trial": {"questions": [{}]}}}, {"id": "1", "cycle": "x"}]})
    assert cards == []


# ─── Podcast ─────────────────────────────────────────────────────────────────────────────


def _course(sections: list[dict[str, Any]]) -> CourseGenerationResponse:
    return CourseGenerationResponse.model_validate({
        "mode": "question_only", "format": "full_course",
        "meta": {"title": "Cours", "subject": "S", "language": "fr", "generated_at": "2026-01-01T00:00:00Z"},
        "sources": [], "introduction": {"quoi": "Intro"}, "sections": sections, "summary": "Résumé",
    })


def _development_text(sections: list[dict[str, Any]]) -> str:
    return next(s.text for s in serialize_course(_course(sections)) if s.kind == "development")


def test_podcast_recall_questions_come_from_the_challenge_and_the_qcm_trial():
    text = _development_text(_stored(cycle_section("Sous-réseaux", _qcm_two_questions)))

    recall = next(line for line in text.splitlines() if line.startswith("Questions de rappel"))
    assert "On découpe 192.168.1.0/24 en 4 parts égales" in recall and "Réponse :" in recall
    assert "Chaque part a 64 adresses" in recall and "Les parts couvrent tout le réseau" in recall
    assert "Combien d'adresses par sous-réseau ?" in recall and "Réponse : 64" in recall
    assert "Quel masque" not in recall  # au plus 2 questions par segment audio
    assert "Combien d'adresses dans un /26" not in recall  # jamais une question numérique


def test_podcast_recall_questions_skip_a_numeric_trial_question_and_an_exercise_card():
    def only_numeric(section: dict) -> None:
        section["cycle"]["trial"]["questions"] = [_numeric_question()]

    for mutate in (only_numeric, _exercise_card):
        text = _development_text(_stored(cycle_section("Sous-réseaux", mutate)))
        recall = next(line for line in text.splitlines() if line.startswith("Questions de rappel"))
        assert "/26" not in recall and "Découpe" not in recall  # seul le défi reste
        assert recall.count("Réponse :") == 1


def test_podcast_still_reads_the_flat_text_of_a_cycle_section():
    text = _development_text(_stored(cycle_section()))

    assert "Un réseau /24 compte 256 adresses" in text  # texte des mini-étapes
    assert "Masque : Indique quelle partie" in text  # vocabulaire


def test_podcast_old_sections_keep_their_check_questions_and_no_cycle_questions():
    section = {
        "id": "0", "title": "ACV", "quoi": "def", "pourquoi": "raison", "comment": "méca",
        "worked_example": {"statement": "", "steps": [], "result": ""},
        "check_questions": [{"question": "Quelle grandeur ?", "options": ["Tension", "Fréquence"], "correct_option_indices": [0]}],
    }

    assert "Questions de rappel : Quelle grandeur ? Réponse : Tension" in _development_text([section])


# ─── Introduction, plan de régénération, assistant, exercices ──────────────────────────────


def test_a_cycle_section_is_never_taken_for_an_introduction():
    (stored,) = _stored(cycle_section())
    assert _is_introduction(stored) is False

    # même vidée de son texte à plat : le cycle suffit à la reconnaître comme section de développement
    flat_less = {**stored, "pourquoi": "", "comment": "", "quoi": ""}
    assert _is_introduction(flat_less) is False


def test_planned_sections_of_a_cycle_course_keep_every_development_section():
    stored = _stored(cycle_section("A"), cycle_section("B"))
    plan = planned_sections_from_course({"sections": [{"id": "9", "title": "Introduction", "quoi": "texte", "pourquoi": "", "comment": ""}, *stored]})

    assert [(p.type, p.title) for p in plan][:3] == [("introduction", "Introduction"), ("development", "A"), ("development", "B")]


def test_assistant_context_reads_the_whole_cycle():
    (stored,) = _stored(cycle_section())
    stored["cycle"]["steps"][1]["widget"] = {"title": "Découpe interactive", "description": "Fais varier le nombre de parts.", "html": "<p>x</p>"}

    text = _section_text(1, stored)

    assert "## Section 1 — Sous-réseaux" in text
    assert "Défi : On découpe 192.168.1.0/24" in text and "(bonne réponse)" in text
    assert "Étape 1 : Un réseau /24 compte 256 adresses" in text and "Étape 4 : Reste à ranger" in text
    assert "Découpe interactive — Fais varier le nombre de parts." in text  # titre et alternative du widget, jamais le HTML
    assert "<p>x</p>" not in text
    assert "Le mot juste :" in text and "- Masque : Indique quelle partie" in text
    assert "Piège classique : " in text and "On croit que le nombre du masque" in text
    assert "Question : Combien d'adresses par sous-réseau ?" in text


def test_assistant_context_of_an_old_section_is_unchanged():
    old = _stored(old_format_section("Ancien"))[0]

    text = _section_text(1, old)

    assert "### Pourquoi" in text and "Défi :" not in text and "Étape 1" not in text


def test_exercises_see_cycle_sections_as_usable_and_use_their_takeaways():
    stored = _stored(cycle_section("Sous-réseaux"))
    course = {"meta": {"title": "Réseaux"}, "sections": stored}

    assert course_is_ready(course) is True
    assert [s["title"] for s in usable_sections(course)] == ["Sous-réseaux"]
    prompt = build_prompt(course, cap=3)
    assert "Sous-réseaux" in prompt and "La taille d'une part se calcule" in prompt  # `key_takeaways` du cycle


def test_an_empty_stored_cycle_is_incomplete_and_not_usable_for_exercises():
    (stored,) = _stored(cycle_section())
    stored["cycle"]["steps"] = []

    assert is_incomplete_section_dict(stored) is True
    assert usable_sections({"sections": [stored]}) == []
