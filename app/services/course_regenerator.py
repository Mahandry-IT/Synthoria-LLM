"""Régénère en entier un cours incomplet déjà persisté, en conservant sa structure.

Le plan d'origine n'est plus disponible (il expire et n'est pas relié au cours) : il est reconstruit
à partir du cours enregistré — mêmes sections, mêmes titres, même ordre — puis rejoué par le
pipeline habituel (`generate_course_from_validated_plan`). Aucun appel de planification n'est donc
dépensé, et les notes de l'apprenant se retrouvent par titre de section.

Seuls les titres survivent : les objectifs et sous-thèmes du plan d'origine ne sont pas stockés avec
le cours, le contenu régénéré peut donc s'en écarter.
"""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import async_sessionmaker

from app.api.schemas import ApiPlannedSection, CourseGenerationResponse
from app.core.config import Settings
from app.db.models import CoursePlan, CourseSession
from app.services.course_depth import depth_of_session
from app.services.course_generator import is_incomplete_section_dict
from app.services.course_plan_generator import build_retrieval_context, generate_course_from_validated_plan
from app.services.gemini_client import GeminiClient
from app.services.vector_store import NumpyVectorStore

_DEVELOPMENT_SUBSECTIONS = {"pourquoi", "quoi", "comment"}
_INTRODUCTION_TITLE = "Introduction"
_CLOSING_SECTIONS = (
    ("common_pitfalls", "Pièges courants"),
    ("summary", "Résumé"),
    ("next_steps", "Pour aller plus loin"),
)


def is_incomplete_course(gemini_response: dict[str, Any]) -> bool:
    """Vrai si le cours a une section incomplète ou s'il lui manque son résumé ou son quiz (cours
    livré sans sa clôture, avant qu'elle ne devienne obligatoire)."""
    if any(is_incomplete_section_dict(s) for s in gemini_response.get("sections") or []):
        return True
    return not (gemini_response.get("summary") or "").strip() or not gemini_response.get("quiz")


def _is_introduction(section: dict[str, Any]) -> bool:
    """Une introduction est stockée sans le découpage Pourquoi/Quoi/Comment : ni sous-section
    typée sous ces titres, ni texte legacy `pourquoi`/`comment` (son texte, à plat, n'occupe que
    `quoi`) — contrairement à toute section de développement, section de repli comprise. Une
    section vide n'en est pas une : c'est une section de développement qu'une régénération ratée
    a vidée."""
    if is_incomplete_section_dict(section):
        return False
    titles = {str(sub.get("title") or "").strip().casefold() for sub in section.get("subsections") or []}
    has_development_text = any((section.get(key) or "").strip() for key in ("pourquoi", "comment"))
    return not (titles & _DEVELOPMENT_SUBSECTIONS) and not has_development_text


def planned_sections_from_course(gemini_response: dict[str, Any]) -> list[ApiPlannedSection]:
    """Plan reconstruit depuis le cours enregistré : son introduction (ou une nouvelle s'il n'en
    a pas), ses sections de développement dans l'ordre, puis pièges, résumé et suite."""
    stored = [s for s in gemini_response.get("sections") or [] if str(s.get("title") or "").strip()]
    introduction_title = _INTRODUCTION_TITLE
    if stored and _is_introduction(stored[0]):
        introduction_title = str(stored[0]["title"]).strip()
        stored = stored[1:]

    planned = [("introduction", introduction_title)]
    planned += [("development", str(s["title"]).strip()) for s in stored]
    planned += list(_CLOSING_SECTIONS)
    return [
        ApiPlannedSection(type=type_, title=title, order=order)
        for order, (type_, title) in enumerate(planned, start=1)
    ]


async def regenerate_course(
    row: CourseSession,
    gemini_client: GeminiClient,
    vector_store: NumpyVectorStore,
    settings: Settings,
    db_session_factory: async_sessionmaker | None = None,
) -> CourseGenerationResponse:
    """Cours régénéré à partir de la question, des fichiers, du mode et des titres du cours `row`.

    Rien n'est écrit ici : l'appelant ne remplace le cours enregistré que si cette fonction réussit.

    Lève: ValueError (aucune section à régénérer), GeminiUnavailableError,
    GeminiQuotaExceededError, GeminiInvalidResponseError.
    """
    stored = row.gemini_response
    sections = planned_sections_from_course(stored)
    if not any(s.type == "development" for s in sections):
        raise ValueError("Ce cours n'a aucune section à régénérer")

    filenames = list(row.filenames or [])
    retrieval_context = await build_retrieval_context(
        row.question, vector_store, gemini_client, settings, mode=row.mode, filename=filenames or None,
    )
    meta = stored.get("meta") or {}
    # Plan transitoire, jamais persisté : seulement le support attendu par le pipeline de génération.
    plan_row = CoursePlan(
        question=row.question,
        mode=row.mode,
        depth=depth_of_session(stored),
        filenames=filenames,
        top_k=settings.course_top_k_default,
        full_document=False,
        retrieval_context=retrieval_context,
        plan={
            "meta": {
                "title": meta.get("title") or row.question,
                "subject": meta.get("subject") or "",
                "language": meta.get("language") or "fr",
            }
        },
        expires_at=datetime.now(timezone.utc),
    )
    return await generate_course_from_validated_plan(
        plan_row, sections, gemini_client, settings, db_session_factory=db_session_factory
    )
