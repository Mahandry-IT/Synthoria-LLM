"""Sérialisation déterministe d'un cours persisté en sections de texte source pour le script.

Fonctions pures. Le cours persisté est déjà « aplati » (les blocs tableau, formule
et code sont devenus du texte) : on nettoie ici ce qui ne doit pas être lu tel quel
(LaTeX brut, blocs de code, Markdown) avant de le confier au modèle de script.
"""

import re
from dataclasses import dataclass
from typing import Any, Literal

from app.api.schemas import CourseAnswer, CourseGenerationResponse, CoursePitfall, CourseSection
from app.services.podcast.tts_normalizer import latex_to_words

SectionKind = Literal["intro", "development", "pitfalls", "summary", "next_steps"]

_FENCED_RE = re.compile(r"```[^\n]*\n?.*?```", re.DOTALL)
_DISPLAY_FORMULA_RE = re.compile(r"\$\$(.+?)\$\$(?:\s*\(([^)]*)\))?", re.DOTALL)
_INLINE_MATH_RE = re.compile(r"\$([^$\n]+?)\$")
_INLINE_CODE_RE = re.compile(r"`([^`]*)`")


@dataclass(frozen=True)
class SourceSection:
    index: int
    title: str
    text: str
    kind: SectionKind

    @property
    def word_count(self) -> int:
        return len(self.text.split())


def _clean(text: str) -> str:
    """Retire le LaTeX brut, les blocs de code et le Markdown d'un texte de cours."""
    text = _FENCED_RE.sub(" (le cours présente ici un extrait de code) ", text)

    def formula(match: re.Match[str]) -> str:
        description = (match.group(2) or "").strip()
        return f" {description} " if description else " (le cours présente ici une formule) "

    text = _DISPLAY_FORMULA_RE.sub(formula, text)
    text = _INLINE_MATH_RE.sub(lambda m: f" {latex_to_words(m.group(1))} ", text)
    text = _INLINE_CODE_RE.sub(r"\1", text)
    text = text.replace("**", "")
    return re.sub(r"[ \t]+", " ", text).strip()


def _join(*parts: tuple[str, str]) -> str:
    return "\n".join(f"{label} : {_clean(value)}" for label, value in parts if value and value.strip())


def _development_text(section: CourseSection | CourseAnswer) -> str:
    example = section.worked_example
    example_text = ""
    if example is not None:
        steps = " ".join(f"Étape {step.id} : {_clean(step.content)}" for step in example.steps)
        example_text = " ".join(
            part for part in (_clean(example.statement), steps, _clean(example.result)) if part
        )
    return _join(
        ("Réponse", getattr(section, "summary", "") or ""),
        ("Quoi", section.quoi or ""),
        ("Pourquoi", section.pourquoi or ""),
        ("Comment", section.comment or ""),
        ("Exemple", example_text),
        ("Points clés", " ; ".join(section.key_points)),
    )


def _pitfalls_text(pitfalls: list[CoursePitfall]) -> str:
    lines = []
    for number, pitfall in enumerate(pitfalls, start=1):
        line = f"Piège {number} : {_clean(pitfall.description)}"
        if pitfall.why_it_happens:
            line += f" Pourquoi : {_clean(pitfall.why_it_happens)}"
        if pitfall.how_to_avoid:
            line += f" Comment l'éviter : {_clean(pitfall.how_to_avoid)}"
        lines.append(line)
    return "\n".join(lines)


def serialize_course(course: CourseGenerationResponse | dict[str, Any]) -> list[SourceSection]:
    """Cours → sections sources ordonnées : introduction, développements, pièges, synthèse, suite.

    Le quiz est volontairement exclu. Les sections sans texte exploitable sont ignorées.
    """
    if isinstance(course, dict):
        course = CourseGenerationResponse.model_validate(course)

    drafts: list[tuple[str, str, SectionKind]] = []

    if course.introduction:
        text = "\n".join(_clean(value) for value in course.introduction.values() if value)
        drafts.append(("Introduction", text, "intro"))

    if course.sections:
        drafts.extend((s.title, _development_text(s), "development") for s in course.sections)
    elif course.answer is not None:
        drafts.append((course.meta.title, _development_text(course.answer), "development"))

    if course.common_pitfalls:
        drafts.append(("Pièges fréquents", _pitfalls_text(course.common_pitfalls), "pitfalls"))
    if course.summary and course.summary.strip():
        drafts.append(("Synthèse", _clean(course.summary), "summary"))
    if course.next_steps:
        drafts.append(("Pour aller plus loin", "\n".join(_clean(s) for s in course.next_steps), "next_steps"))

    usable = [(title, text, kind) for title, text, kind in drafts if text.strip()]
    return [SourceSection(index=i, title=title, text=text, kind=kind) for i, (title, text, kind) in enumerate(usable)]


def has_usable_content(sections: list[SourceSection]) -> bool:
    """Vrai s'il existe au moins une section de développement ou d'introduction non vide."""
    return any(s.kind in ("intro", "development") and s.text.strip() for s in sections)
