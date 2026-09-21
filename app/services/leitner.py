"""Répétition espacée (système de Leitner) et extraction des flashcards d'un cours."""

from datetime import datetime, timedelta
from typing import Any


def next_review(box: int | None, correct: bool, now: datetime, intervals_days: list[int]) -> tuple[int, datetime]:
    """Boîte et échéance après une révision (`box` None = carte jamais révisée).

    Succès : première boîte pour une carte nouvelle, sinon boîte suivante (plafonnée à la dernière) ;
    échec : retour à la première boîte. L'échéance est `now` + l'intervalle de la nouvelle boîte
    (J+1, J+3, J+7, J+21 par défaut).
    """
    last = len(intervals_days) - 1
    new_box = (0 if box is None else min(max(box, 0) + 1, last)) if correct else 0
    return new_box, now + timedelta(days=intervals_days[new_box])


def flashcards_from_course(course: dict[str, Any]) -> list[dict[str, Any]]:
    """Cartes dérivées des questions « Vérifie » de chaque section d'un cours (dict de session).

    `card_id` = « <id de section>-<n° de question> » : stable tant que le cours ne change pas.
    Recto = la question ; verso = la (ou les) bonne(s) réponse(s) suivie(s) de l'explication.
    """
    cards: list[dict[str, Any]] = []
    for section in course.get("sections") or []:
        for index, q in enumerate(section.get("check_questions") or []):
            options = q.get("options") or []
            answers = [options[i] for i in q.get("correct_option_indices", []) if 0 <= i < len(options)]
            if not answers:
                continue
            back = " ; ".join(answers)
            if q.get("explanation"):
                back = f"{back}\n{q['explanation']}"
            cards.append(
                {
                    "card_id": f"{section.get('id')}-{index}",
                    "front": q.get("question", ""),
                    "back": back,
                    "section_ref": section.get("id"),
                }
            )
    return cards
