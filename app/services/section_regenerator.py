"""Régénère le contenu d'UNE section « incomplète » d'un cours déjà persisté.

Contrairement à la génération complète du cours (best-effort, replis silencieux), cette
régénération ciblée lève toujours en cas d'échec : l'apprenant doit savoir que sa tentative
n'a pas abouti plutôt que de se voir renvoyer silencieusement l'ancienne section incomplète.
"""

from pydantic import ValidationError

from app.core.config import Settings
from app.core.exceptions import GeminiInvalidResponseError
from app.db.models import CourseSession
from app.schemas.course_generation import Section, SectionType, SectionsBatchSchema
from app.services.course_generator import _build_context_block, _get_teacher_instructions, _retrieve_chunks
from app.services.gemini_client import GeminiClient
from app.services.vector_store import NumpyVectorStore


async def _context_block(
    row: CourseSession, vector_store: NumpyVectorStore, gemini_client: GeminiClient, settings: Settings
) -> str:
    """Contexte source pour régénérer une section : même origine que la génération initiale de la session."""
    if row.mode == "question_only":
        raw_answer, web_sources = await gemini_client.search_grounded(
            prompt=f"Question de l'utilisateur : {row.question}",
            system_instruction=_get_teacher_instructions(),
        )
        return f"Sources web disponibles : {web_sources}\n\nSynthèse :\n{raw_answer}"
    _, chunks = await _retrieve_chunks(
        row.question, vector_store, gemini_client, settings, row.mode, None, row.filenames or None, False,
    )
    return _build_context_block(chunks)


async def regenerate_section(
    row: CourseSession,
    section_title: str,
    gemini_client: GeminiClient,
    vector_store: NumpyVectorStore,
    settings: Settings,
) -> Section:
    """Régénère une section DEVELOPMENT complète (cycle pédagogique inclus), par son titre.

    Lève: GeminiUnavailableError, GeminiQuotaExceededError, GeminiInvalidResponseError.
    """
    system_instruction = _get_teacher_instructions()
    context_block = await _context_block(row, vector_store, gemini_client, settings)

    prompt = (
        f"Question de l'utilisateur (contexte du cours) : {row.question}\n\n"
        f"Contexte source :\n{context_block}\n\n"
        f"Génère UNE SEULE section DEVELOPMENT intitulée exactement « {section_title} », complète : "
        "sous-sections Pourquoi/Quoi/Comment (avec un exemple travaillé complet dans Comment), un défi "
        "(`challenge`), un exemple à trous (`faded_example`), 2-3 `check_questions` et un `recall_prompt`. "
        "Retourne le JSON selon le schéma fourni."
    )
    structured = await gemini_client.format_structured(
        raw_answer=prompt, system_instruction=system_instruction, response_schema=SectionsBatchSchema,
    )
    try:
        parsed = SectionsBatchSchema.model_validate(structured)
    except ValidationError as exc:
        raise GeminiInvalidResponseError(f"Section régénérée invalide: {exc}") from exc
    if not parsed.sections:
        raise GeminiInvalidResponseError("Aucune section n'a été renvoyée")

    return parsed.sections[0].model_copy(update={"type": SectionType.DEVELOPMENT, "title": section_title})
