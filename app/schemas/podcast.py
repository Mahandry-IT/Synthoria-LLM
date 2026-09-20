"""Schémas Pydantic du script de podcast (structured output Gemini).

Volontairement **plats** (pas d'union ni de récursion) pour rester compatibles
avec `response_json_schema`. Les champs déterministes (voix, pauses, durées,
budget de mots) sont calculés en code et ne figurent pas ici.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Speaker = Literal["HOST", "EXPERT"]


class PodcastTurn(BaseModel):
    speaker: Speaker = Field(description="HOST pose les questions et relance ; EXPERT explique.")
    text: str = Field(
        min_length=1,
        description=(
            "Réplique en français oral : phrases courtes, aucun symbole, aucune formule, "
            "aucun code, aucun Markdown, nombres et unités écrits comme ils se prononcent."
        ),
    )


class PodcastSegment(BaseModel):
    section_ref: int = Field(
        ge=0,
        description="Index (0-based) de la section source du cours que ce segment traite.",
    )
    title: str = Field(min_length=1, description="Titre court du segment (chapitre audio).")
    turns: list[PodcastTurn] = Field(min_length=1, description="Répliques du segment, dans l'ordre.")


class PodcastSegmentsBatch(BaseModel):
    """Segments générés pour un lot de sections du cours."""

    segments: list[PodcastSegment] = Field(
        description="Exactement un segment par section source du lot, dans le même ordre."
    )


class PodcastFrame(BaseModel):
    """Titre, introduction et conclusion du podcast."""

    title: str = Field(min_length=1, description="Titre du podcast.")
    intro_turns: list[PodcastTurn] = Field(min_length=1, description="Accroche et annonce du plan.")
    outro_turns: list[PodcastTurn] = Field(min_length=1, description="Récapitulatif et conclusion.")


class PodcastScript(BaseModel):
    title: str
    intro_turns: list[PodcastTurn]
    segments: list[PodcastSegment]
    outro_turns: list[PodcastTurn]
