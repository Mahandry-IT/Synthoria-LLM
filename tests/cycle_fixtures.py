"""Section DEVELOPMENT au format « cycle » conforme au mode standard, partagée par les tests du cycle.

`cycle_section()` renvoie le dict d'une section (avant validation Pydantic) : 4 mini-étapes aux supports
tous différents, une prédiction `choice`, 3 termes, 2 pièges `spot`, une épreuve `qcm`. Aucun problème
bloquant ni de forme. `mutate` permet à chaque test de casser UN élément, sur une copie.
"""

import copy
from collections.abc import Callable
from typing import Any

from tests.lesson_box_fixtures import BOXES_FIELDS


def _quiz_question() -> dict[str, Any]:
    return {
        "question": "Combien d'adresses par sous-réseau ?",
        "kind": "choice",
        "choices": ["32", "64", "128"],
        "correct_indices": [1],
        "difficulty": "normale",
        "explanation": "256 divisé par 4 donne 64.",
        "explanation_per_choice": ["Trop peu.", "Exact.", "Trop."],
        "requires_calculation": True,
    }


def cycle_dict() -> dict[str, Any]:
    return {
        "thread": "Le réseau 192.168.1.0/24 découpé en 4 sous-réseaux égaux.",
        "challenge": {
            "question": "On découpe 192.168.1.0/24 en 4 parts égales. Quelles affirmations sont vraies ?",
            "options": ["Chaque part est plus petite", "Chaque part a 64 adresses", "Les parts se chevauchent", "Les parts couvrent tout le réseau"],
            "correct_indices": [1, 3],
            "feedback": "Quatre parts égales de 256 adresses font 64 adresses chacune.",
        },
        "steps": [
            {
                "text": "Un réseau /24 compte 256 adresses. On veut quatre parts égales.",
                "action": "Classe chaque adresse dans son sous-réseau.",
                "support": {
                    "kind": "classify", "caption": "Où va chaque adresse ?",
                    "categories": ["Sous-réseau A", "Sous-réseau B"],
                    "cases": [
                        {"label": "192.168.1.10", "category_index": 0},
                        {"label": "192.168.1.70", "category_index": 1},
                        {"label": "192.168.1.20", "category_index": 0},
                    ],
                },
                "prediction": None,
            },
            {
                "text": "On partage 256 adresses en 4 parts égales.",
                "action": "Avance étape par étape dans le graphique.",
                "support": {
                    "kind": "chart", "caption": "Adresses par part",
                    "chart": {
                        "kind": "bar", "caption": "Adresses par part", "labels": ["A", "B", "C", "D"],
                        "series": [{"name": "Adresses", "values": [64.0, 64.0, 64.0, 64.0]}],
                        "steps": [{"caption": "Première part", "highlight_labels": [0]}, {"caption": "Toutes les parts", "highlight_labels": [0, 1, 2, 3]}],
                    },
                },
                "prediction": {
                    "kind": "choice", "question": "Combien d'adresses par part ?",
                    "options": ["32", "64", "128"], "correct_index": 1,
                    "outcome": "En réalité, 64 adresses par part.",
                },
            },
            {
                "text": "Les réseaux commencent aux multiples de la taille d'une part. Il y en a quatre.",
                "action": "Complète la case vide.",
                "support": {
                    "kind": "fill_table", "caption": "Les débuts",
                    "table": {"caption": "Plages", "headers": ["Réseau", "Début", "Fin"], "rows": [["A", "0", "63"], ["B", "64", "127"], ["C", "128", "191"]]},
                    "blanks": [{"row": 1, "col": 1}],
                },
                "prediction": None,
            },
            {
                "text": "Reste à ranger les étapes du calcul.",
                "action": "Remets les étapes dans l'ordre.",
                "support": {
                    "kind": "order", "caption": "Le calcul",
                    "order_items": ["Compter les adresses", "Diviser par quatre", "Lister les débuts"],
                    "order_criterion": "du premier au dernier geste",
                },
                "prediction": None,
            },
        ],
        "vocabulary": [
            {"term": "Masque", "plain": "Indique quelle partie de l'adresse désigne le réseau."},
            {"term": "Sous-réseau", "plain": "Une part du réseau de départ."},
            {"term": "Plage", "plain": "Les adresses d'un début à une fin."},
        ],
        "traps": {
            "mode": "spot",
            "entries": [
                {
                    "statements": ["Un /26 compte 64 adresses", "Un /26 compte 26 adresses", "Un /26 est plus petit qu'un /24"],
                    "wrong_index": 1,
                    "explanation": "On croit que le nombre du masque est le nombre d'adresses ; c'est le nombre de bits du réseau.",
                },
                {
                    "statements": ["Deux parts égales ne se chevauchent pas", "Un sous-réseau peut dépasser le réseau", "Chaque part garde la même taille"],
                    "wrong_index": 1,
                    "explanation": "On croit qu'on peut empiéter ; les parts restent dans le réseau de départ.",
                },
            ],
        },
        "trial": {"kind": "qcm", "questions": [_quiz_question()]},
    }


def cycle_section(title: str = "Sous-réseaux", mutate: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Dict de section prêt pour `Section.model_validate`. `mutate(section)` casse un élément (sur une copie)."""
    section: dict[str, Any] = {
        "type": "development", "title": title,
        "cycle": cycle_dict(),
        "widget_brief": {"step_number": 2, "brief": "Curseur du nombre de parts : la taille de chaque part se met à jour."},
        "recall_prompt": {"prompt": "Explique avec tes mots comment on découpe un réseau.", "expected_key_points": ["parts égales", "taille d'une part"]},
        "key_takeaways": ["Un réseau se découpe en parts égales", "La taille d'une part se calcule", "Les débuts sont des multiples"],
        "covered_subtopics": [],
    }
    section = copy.deepcopy(section)
    if mutate is not None:
        mutate(section)
    return section


def old_format_section(title: str = "Ancien") -> dict[str, Any]:
    """Section de l'ancien format (sans `cycle`), conforme au mode standard."""
    return {
        "type": "development", "title": title, "blocks": [], **BOXES_FIELDS,
        "subsections": [
            {"title": "Pourquoi", "blocks": [{"type": "list", "list_items": ["enjeu"]}]},
            {"title": "Quoi", "blocks": [{"type": "text", "text": "définition"}, {"type": "table", "table": {"caption": "c", "headers": ["h"], "rows": [["x"]]}}]},
            {"title": "Comment", "blocks": [{"type": "text", "text": "mécanisme"}]},
        ],
    }
