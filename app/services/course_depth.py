"""Modes de cours (`express` / `standard` / `approfondi`) : point de vérité unique des profils.

Prompts (plan, lots, réparation, régénération, ajout de sections, quiz) et validateur
(`visual_validation.visual_issues`) lisent leurs chiffres ici, jamais dans les `.md` d'instructions.

Rétro-compatibilité : un plan ou une session sans mode (données antérieures) est traité comme
`approfondi` (`LEGACY_DEPTH`), quel que soit le défaut configuré pour les nouvelles requêtes.
"""

from dataclasses import dataclass
from typing import Any, Literal, get_args

CourseDepth = Literal["express", "standard", "approfondi"]
COURSE_DEPTHS: tuple[str, ...] = get_args(CourseDepth)
# Valeur des données sans mode (plans/sessions antérieurs) : elles restent générées comme avant.
LEGACY_DEPTH: CourseDepth = "approfondi"


@dataclass(frozen=True)
class DepthProfile:
    """Profil d'un mode de cours.

    `max_sections` à None : nombre de sections piloté par la couverture (plancher seulement).
    `quiz_open_ended` : la borne haute du quiz est indicative (« 10-12+ »), jamais un plafond.
    """

    name: CourseDepth
    min_sections: int
    max_sections: int | None
    non_text_ratio: float
    max_prose_words: int
    max_blocks: int
    check_questions_min: int
    check_questions_max: int
    quiz_min: int
    quiz_max: int
    quiz_open_ended: bool = False


PROFILES: dict[str, DepthProfile] = {
    "express": DepthProfile(
        name="express", min_sections=7, max_sections=12, non_text_ratio=0.5,
        max_prose_words=150, max_blocks=5, check_questions_min=1, check_questions_max=2,
        quiz_min=5, quiz_max=6,
    ),
    "standard": DepthProfile(
        name="standard", min_sections=13, max_sections=16, non_text_ratio=0.5,
        max_prose_words=300, max_blocks=8, check_questions_min=2, check_questions_max=3,
        quiz_min=8, quiz_max=10,
    ),
    "approfondi": DepthProfile(
        name="approfondi", min_sections=18, max_sections=None, non_text_ratio=0.5,
        max_prose_words=500, max_blocks=12, check_questions_min=2, check_questions_max=3,
        quiz_min=10, quiz_max=12, quiz_open_ended=True,
    ),
}


def get_profile(depth: str | None) -> DepthProfile:
    """Profil du mode `depth` ; absent ou inconnu (données antérieures) → `LEGACY_DEPTH`."""
    return PROFILES[normalize_depth(depth)]


def normalize_depth(value: Any) -> CourseDepth:
    """Mode valide, sinon `LEGACY_DEPTH` (valeur absente, inconnue ou d'un ancien enregistrement)."""
    return value if isinstance(value, str) and value in COURSE_DEPTHS else LEGACY_DEPTH


def depth_of_plan(plan_row: Any) -> CourseDepth:
    """Mode d'un plan persisté (colonne `depth`, migration 015) ; absent → `LEGACY_DEPTH`."""
    return normalize_depth(getattr(plan_row, "depth", None))


def depth_of_session(gemini_response: dict[str, Any] | None) -> CourseDepth:
    """Mode d'une session persistée (`meta.depth` du JSONB) ; absent → `LEGACY_DEPTH`."""
    meta = (gemini_response or {}).get("meta") or {}
    return normalize_depth(meta.get("depth") if isinstance(meta, dict) else None)


def _range(low: int, high: int) -> str:
    return str(low) if low == high else f"{low} à {high}"


def render_plan_rules(profile: DepthProfile) -> str:
    """Consigne de nombre de sections `development`, injectée dans les prompts de plan."""
    if profile.max_sections is None:
        count = (
            f"au moins {profile.min_sections} sections `development` (18-20 pour un sujet simple, 25-40+ pour un "
            "sujet large ou un document volumineux) : un plancher, jamais un plafond — le nombre est piloté par la couverture complète du sujet"
        )
    else:
        count = (
            f"entre {profile.min_sections} et {profile.max_sections} sections `development` (plafond strict) : "
            "ne garde que l'essentiel du sujet ; chaque section porte 3 à 5 sous-thèmes"
        )
    return f"--- Mode du cours : {profile.name} ---\nNombre de sections : {count}."


def render_rules(profile: DepthProfile) -> str:
    """Règles de forme d'une section `development` (ratio, budget, questions), injectées dans les
    prompts de génération, de réparation, de régénération et d'ajout de sections."""
    return (
        f"--- Règles du mode « {profile.name} » pour chaque section DEVELOPMENT ---\n"
        f"- Au moins {profile.non_text_ratio:.0%} des blocs sont non textuels (TABLE, LIST, DIAGRAM, CHART, "
        "FORMULA, CODE, WORKED_EXAMPLE...) : TEXT, DEFINITION et CALLOUT sont textuels.\n"
        f"- Au plus {profile.max_prose_words} mots de prose par section (TEXT, DEFINITION, CALLOUT, éléments "
        "de LIST et exemple travaillé) ; tableaux, formules, code, schémas et graphiques ne comptent pas dans "
        "les mots.\n"
        f"- Au plus {profile.max_blocks} blocs par section, toutes sous-sections confondues.\n"
        "- Chaque bloc TEXT tient en 3 phrases au plus.\n"
        f"- {_range(profile.check_questions_min, profile.check_questions_max)} `check_questions` par section.\n"
        "- Renseigne `challenge_key_points` : 2 à 4 idées courtes qu'une bonne réponse au défi contient.\n"
        "- Arbitrage : si couvrir tous les sous-thèmes du plan oblige à dépasser le budget, la couverture "
        "gagne — condense la prose (tableaux, listes) plutôt que d'omettre un sous-thème."
    )


def render_quiz_rules(profile: DepthProfile) -> str:
    """Consigne de taille du quiz final, injectée dans le prompt de clôture du cours."""
    size = (
        f"au moins {profile.quiz_min} à {profile.quiz_max} questions (plancher, environ 1 à 2 par section "
        "DEVELOPMENT si le cours en compte davantage)"
        if profile.quiz_open_ended
        else f"{profile.quiz_min} à {profile.quiz_max} questions"
    )
    return f"--- Quiz final (mode « {profile.name} ») : {size}. ---"


def render_course_rules(profile: DepthProfile) -> str:
    """Toutes les règles du mode, pour la génération directe du cours (sans plan)."""
    return "\n".join([render_plan_rules(profile), render_rules(profile), render_quiz_rules(profile)])
