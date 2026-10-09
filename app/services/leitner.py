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


def card_back(choices: list[str], correct_indices: list[int], explanation: str) -> str:
    """Verso d'une carte : la (ou les) bonne(s) réponse(s) suivie(s) de l'explication."""
    back = " ; ".join(choices[i] for i in correct_indices if 0 <= i < len(choices))
    return f"{back}\n{explanation}" if explanation else back


def review_mode(box: int, variant_no: int, has_choices: bool) -> str:
    """Mode d'affichage déterministe : QCM si `box + variant_no` est pair, réponse libre sinon ;
    une carte sans choix est toujours en réponse libre."""
    return "qcm" if has_choices and (box + variant_no) % 2 == 0 else "text"


def resolve_variant_no(stored: int, available: set[int]) -> int:
    """Variante réellement affichable : la plus haute existante ≤ `stored` (la 0, dérivée des
    `check_questions`, existe toujours). Tant que la variante suivante n'est pas générée, la
    carte retombe ainsi sur la variante courante."""
    return max((n for n in available | {0} if n <= max(stored, 0)), default=0)


def flashcards_from_course(course: dict[str, Any]) -> list[dict[str, Any]]:
    """Cartes dérivées des questions « Vérifie » de chaque section d'un cours (dict de session).

    `card_id` = « <id de section>-<n° de question> » : stable tant que le cours ne change pas.
    Recto = la question ; verso = la (ou les) bonne(s) réponse(s) suivie(s) de l'explication.
    `choices`, `correct_indices` et `explanation` servent au mode QCM de la révision (variante 0).
    """
    cards: list[dict[str, Any]] = []
    for section in course.get("sections") or []:
        for index, q in enumerate(section.get("check_questions") or []):
            options = q.get("options") or []
            correct = [i for i in q.get("correct_option_indices", []) if 0 <= i < len(options)]
            if not correct:
                continue
            explanation = q.get("explanation") or ""
            cards.append(
                {
                    "card_id": f"{section.get('id')}-{index}",
                    "front": q.get("question", ""),
                    "back": card_back(options, correct, explanation),
                    "section_ref": section.get("id"),
                    "choices": options,
                    "correct_indices": correct,
                    "explanation": explanation,
                }
            )
    return cards
