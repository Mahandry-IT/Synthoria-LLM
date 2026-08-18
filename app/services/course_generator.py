import logging
from pathlib import Path
from typing import Any

from app.api.schemas import CourseGenerationResponse
from app.core.config import Settings
from app.core.exceptions import GeminiInvalidResponseError
from app.services.gemini_client import GeminiClient
from app.services.vector_store import NumpyVectorStore

logger = logging.getLogger(__name__)

_DEFAULT_INSTRUCTIONS = (
    "Tu es un professeur pédagogue. Réponds en français avec la méthode "
    "Quoi/Pourquoi/Comment, en incluant systématiquement un exemple travaillé complet."
)
_teacher_instructions_cache: str | None = None


def _load_teacher_instructions() -> str:
    candidates = [
        Path(__file__).resolve().parents[2] / "instruction" / "course_generation_instructions.md",
        Path(__file__).resolve().parents[1] / "instruction" / "course_generation_instructions.md",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    logger.warning("course_generation_instructions_missing_fallback_to_default")
    return _DEFAULT_INSTRUCTIONS


def _get_teacher_instructions() -> str:
    global _teacher_instructions_cache
    if _teacher_instructions_cache is None:
        _teacher_instructions_cache = _load_teacher_instructions()
    return _teacher_instructions_cache


def _filter_by_filename(chunks: list[dict[str, Any]], filename: str | None) -> list[dict[str, Any]]:
    """Filtre les chunks retrouvés sur un nom de fichier (post-filtrage, le store n'a pas d'index natif)."""
    if not filename:
        return chunks
    return [c for c in chunks if c.get("metadata", {}).get("filename") == filename]


def _build_context_block(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return "Aucun extrait de fichier pertinent trouvé pour cette question."
    blocks = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        origin = f"{meta.get('filename', 'inconnu')} (page {meta.get('page', '?')})"
        blocks.append(f"[Source fichier: {origin}]\n{chunk['content']}")
    return "\n\n".join(blocks)


def _file_sources_from_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "type": "file",
            "label": chunk.get("metadata", {}).get("filename", "document"),
            "reference": f"page {chunk.get('metadata', {}).get('page', '?')}",
        }
        for chunk in chunks
    ]


async def generate_course_from_question(
    question: str,
    vector_store: NumpyVectorStore,
    gemini_client: GeminiClient,
    settings: Settings,
    mode: str = "file_question",
    top_k: int | None = None,
    filename: str | None = None,
) -> CourseGenerationResponse:
    """
    Orchestration RAG + génération de cours structuré.

    Paramètres:
        question: question de l'utilisateur.
        vector_store: store vectoriel local (retrieval top-k).
        gemini_client: client Gemini (2 appels).
        settings: configuration applicative.
        mode: "file_question" (Mode 2) ou "question_only" (Mode 3, préparé mais non branché).
        top_k: nombre de chunks à récupérer (défaut settings.course_top_k_default).
        filename: filtre optionnel sur un document déjà ingéré.

    Retour: CourseGenerationResponse validé.

    Fonctionnement:
        1. Retrieval des chunks pertinents dans le vector store local (sauf mode question_only).
        2. Appel 1 Gemini (Flash + google_search) : réponse groundée (contexte fichier + web).
        3. Appel 2 Gemini (Flash-Lite + response_schema) : mise en forme JSON stricte.

    Lève: GeminiUnavailableError, GeminiQuotaExceededError, GeminiInvalidResponseError.
    """
    resolved_top_k = top_k or settings.course_top_k_default

    chunks: list[dict[str, Any]] = []
    if mode != "question_only":
        raw_chunks = await vector_store.search(question, top_k=resolved_top_k)
        chunks = _filter_by_filename(raw_chunks, filename)

    context_block = _build_context_block(chunks)
    file_sources = _file_sources_from_chunks(chunks)
    system_instruction = _get_teacher_instructions()

    prompt = (
        f"Question de l'utilisateur : {question}\n\n"
        f"Contexte extrait des documents fournis (à compléter par une recherche web si nécessaire) :\n"
        f"{context_block}"
    )

    raw_answer, web_sources = await gemini_client.search_grounded(
        prompt=prompt,
        system_instruction=system_instruction,
    )

    formatting_prompt = (
        f'mode="{mode}"\n'
        f"Sources fichier disponibles : {file_sources}\n"
        f"Sources web disponibles : {web_sources}\n\n"
        f"Réponse brute à structurer en JSON selon le schéma fourni :\n{raw_answer}"
    )

    structured = await gemini_client.format_structured(
        raw_answer=formatting_prompt,
        response_schema=CourseGenerationResponse,
        system_instruction=system_instruction,
    )

    try:
        return CourseGenerationResponse.model_validate(structured)
    except Exception as exc:
        raise GeminiInvalidResponseError(f"JSON structuré invalide: {exc}") from exc