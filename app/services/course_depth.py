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


CycleTrapMode = Literal["cards", "spot", "spot_rewrite"]


@dataclass(frozen=True)
class CycleProfile:
    """Règles du « cycle » (Défi, Sous le capot, Le mot juste, Les pièges classiques, Épreuve) d'un mode.

    Un profil existe pour chaque mode, mais il n'est utilisé que si `DepthProfile.use_cycle` est vrai :
    standard et approfondi (express : profil posé d'après le cahier des charges, jamais activé)."""

    steps_min: int  # mini-étapes de « Sous le capot » acceptées par la validation
    steps_max: int
    steps_target: int  # nombre visé dans le prompt
    steps_ref: int  # nombre « normal » : référence de l'ajustement de durée
    predictions_min: int
    predictions_max: int
    traps_mode: CycleTrapMode
    traps_min: int
    traps_max: int
    vocab_min: int
    vocab_max: int
    minutes_min: int  # durée de base, en minutes (avant ajustement par le nombre de mini-étapes)
    minutes_max: int
    takeaways: int


@dataclass(frozen=True)
class DepthProfile:
    """Profil d'un mode de cours.

    `max_sections` à None : nombre de sections piloté par la couverture (plancher seulement).
    `quiz_open_ended` : la borne haute du quiz est indicative (« 10-12+ »), jamais un plafond.
    `use_cycle` : les sections de développement de ce mode suivent le format « cycle » (objet `cycle`) au
    lieu du format Pourquoi / Quoi / Comment. Un mode s'y convertit en passant ce réglage à True.
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
    takeaways_min: int
    takeaways_max: int
    pitfalls_min: int
    pitfalls_max: int
    quiz_open_ended: bool = False
    use_cycle: bool = False
    cycle: CycleProfile | None = None


PROFILES: dict[str, DepthProfile] = {
    "express": DepthProfile(
        name="express", min_sections=7, max_sections=12, non_text_ratio=0.5,
        max_prose_words=150, max_blocks=5, check_questions_min=1, check_questions_max=2,
        quiz_min=5, quiz_max=6,
        takeaways_min=2, takeaways_max=2, pitfalls_min=1, pitfalls_max=1,
        cycle=CycleProfile(
            steps_min=3, steps_max=3, steps_target=3, steps_ref=3, predictions_min=1, predictions_max=1,
            traps_mode="cards", traps_min=2, traps_max=2, vocab_min=2, vocab_max=4,
            minutes_min=3, minutes_max=3, takeaways=2,
        ),
    ),
    "standard": DepthProfile(
        name="standard", min_sections=13, max_sections=16, non_text_ratio=0.5,
        max_prose_words=300, max_blocks=8, check_questions_min=2, check_questions_max=3,
        quiz_min=8, quiz_max=10,
        takeaways_min=3, takeaways_max=3, pitfalls_min=1, pitfalls_max=1,
        use_cycle=True,
        cycle=CycleProfile(
            steps_min=3, steps_max=5, steps_target=4, steps_ref=4, predictions_min=1, predictions_max=2,
            traps_mode="spot", traps_min=2, traps_max=3, vocab_min=3, vocab_max=6,
            minutes_min=5, minutes_max=5, takeaways=3,
        ),
    ),
    "approfondi": DepthProfile(
        name="approfondi", min_sections=18, max_sections=None, non_text_ratio=0.5,
        max_prose_words=500, max_blocks=12, check_questions_min=2, check_questions_max=3,
        quiz_min=10, quiz_max=12, quiz_open_ended=True,
        takeaways_min=3, takeaways_max=4, pitfalls_min=2, pitfalls_max=2,
        use_cycle=True,
        cycle=CycleProfile(
            steps_min=5, steps_max=6, steps_target=5, steps_ref=5, predictions_min=2, predictions_max=2,
            traps_mode="spot_rewrite", traps_min=2, traps_max=3, vocab_min=3, vocab_max=8,
            minutes_min=8, minutes_max=10, takeaways=3,
        ),
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
        f"- {_range(profile.takeaways_min, profile.takeaways_max)} `key_takeaways` (puces courtes : l'essentiel à "
        f"retenir) et {_range(profile.pitfalls_min, profile.pitfalls_max)} `pitfalls_to_avoid` "
        "(`mistake` + `how_to_avoid`) par section.\n"
        "- Renseigne `challenge_key_points` : 2 à 4 idées courtes qu'une bonne réponse au défi contient.\n"
        "- Arbitrage : si couvrir tous les sous-thèmes du plan oblige à dépasser le budget, la couverture "
        "gagne — condense la prose (tableaux, listes) plutôt que d'omettre un sous-thème."
    )


def cycle_profile_of(profile: DepthProfile) -> CycleProfile:
    """Règles du cycle du mode (toujours définies, même quand `use_cycle` est faux)."""
    assert profile.cycle is not None, f"profil {profile.name} sans règles de cycle"
    return profile.cycle


CYCLE_MINUTES_CAP = (1, 30)
# Une mini-étape de plus ou de moins que la référence du mode change la durée d'autant (en minutes).
_MINUTES_PER_STEP = 1


def estimate_minutes(depth: str, steps: int) -> tuple[int, int]:
    """Durée estimée `(minutes_min, minutes_max)` d'une section au format cycle, calculée côté serveur.

    Base du mode (express 3, standard 5, approfondi 8 à 10 min), ajustée de 1 minute par mini-étape d'écart
    avec le nombre de référence du mode, bornée à 1-30. Aucun appel Gemini : la durée n'est jamais demandée
    au modèle."""
    c = cycle_profile_of(get_profile(depth))
    shift = (steps - c.steps_ref) * _MINUTES_PER_STEP
    low_cap, high_cap = CYCLE_MINUTES_CAP
    low = min(max(c.minutes_min + shift, low_cap), high_cap)
    high = min(max(c.minutes_max + shift, low), high_cap)
    return low, high


def render_cycle_rules(profile: DepthProfile) -> str:
    """Règles de forme du `cycle` d'une section `development`, injectées dans les prompts de génération et
    de réparation des modes qui utilisent le cycle. Les chiffres ne vivent qu'ici, jamais dans les `.md`."""
    c = cycle_profile_of(profile)
    steps = (
        _range(c.steps_min, c.steps_max) + (f" (vise {c.steps_target})" if c.steps_min != c.steps_max else "")
    )
    traps = {
        "cards": "`cards` (cartes à révéler : `claim` + `explanation`)",
        "spot": "`spot` (repérer l'idée fausse : `statements` 3 à 4 + `wrong_index` + `explanation`)",
        "spot_rewrite": (
            "`spot_rewrite` (repérer l'idée fausse PUIS choisir la bonne formulation : `statements`, "
            "`wrong_index`, `rewrite_options`, `correct_rewrite_index`, `explanation`)"
        ),
    }[c.traps_mode]
    return (
        f"--- Règles du mode « {profile.name} » pour le `cycle` de chaque section DEVELOPMENT ---\n"
        f"- `steps` (Sous le capot) : {steps} mini-étapes, chacune avec son support.\n"
        f"- Prédictions : {_range(c.predictions_min, c.predictions_max)} dans toute la section (au plus une par "
        "mini-étape, `prediction` à null ailleurs).\n"
        f"- `traps.mode` = {traps}, {_range(c.traps_min, c.traps_max)} éléments dans `traps.entries`.\n"
        f"- `vocabulary` : {_range(c.vocab_min, c.vocab_max)} termes, uniquement ceux rencontrés dans les mini-étapes.\n"
        "- `trial` : `qcm` (1 à 2 questions) ou `exercise_card` selon la nature de la notion ; un calcul ou une "
        "procédure → `exercise_card`.\n"
        f"- `key_takeaways` : {c.takeaways} puces courtes ; `recall_prompt` renseigné.\n"
        "- Pas de champ de durée à produire : elle est calculée côté serveur.\n"
        "- Les champs de l'ancien format (`subsections`, `blocks`, `challenge`, `faded_example`, `check_questions`, "
        "`pitfalls_to_avoid`) ne sont PAS produits pour ces sections."
    )


CYCLE_CONTENT_DEMAND = (
    "son objet `cycle` complet (`thread`, `challenge`, `steps` avec le support de chaque mini-étape, "
    "`vocabulary`, `traps`, `trial`), au moins une mini-étape de reconstruction (classer, ordonner, compléter un "
    "tableau ou relier), un `widget_brief` (une phrase pour le visuel simulable, seulement s'il existe un vrai "
    "mécanisme à faire varier), un `recall_prompt` et les `key_takeaways`"
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
