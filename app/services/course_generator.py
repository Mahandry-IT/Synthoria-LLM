import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.api.schemas import CourseGenerationResponse, CourseMeta, CourseSource
from app.core.config import Settings
from app.core.exceptions import GeminiInvalidResponseError
from app.schemas.course_generation import CourseGenerationSchema
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


def _block_to_text(block: Any) -> str:
    """Sérialise un ContentBlock en texte plat pour l'API legacy (quoi/pourquoi/comment).

    Avant ce correctif, seuls les blocks TEXT étaient lus — un block TABLE ou
    FORMULA généré par Gemini juste après une phrase d'intro disparaissait
    silencieusement de la réponse API (aucune erreur, contenu juste absent).
    """
    if block.text:
        return block.text
    if block.table:
        rows = "; ".join(" | ".join(row) for row in block.table.rows)
        return f"{' | '.join(block.table.headers)} — {rows}"
    if block.formula:
        rendered = f"$${block.formula.latex}$$"
        return rendered + (f" ({block.formula.description})" if block.formula.description else "")
    if block.list_items:
        return " ; ".join(block.list_items)
    if block.code:
        return block.code
    return ""


def _map_schema_to_response(schema: CourseGenerationSchema) -> CourseGenerationResponse:
    """Convertit la réponse Gemini (CourseGenerationSchema) en CourseGenerationResponse API.

    Le schema Gemini est block-based (sections > subsections > ContentBlocks).
    On extrait les données pertinentes pour le format API existant.
    """
    _SOURCE_MAP = {"file_chunk": "file", "web": "web"}
    all_sources = [
        CourseSource(
            type=_SOURCE_MAP.get(s.type.value, s.type.value),
            label=s.label,
            reference=s.reference,
        )
        for s in schema.sources
    ]

    # Extraire summary et next_steps depuis les sections
    summary = ""
    next_steps: list[str] = []
    for section in schema.sections:
        if section.type.value == "summary":
            for block in section.blocks:
                if block.text:
                    summary = block.text
                    break
        elif section.type.value == "next_steps":
            for block in section.blocks:
                if block.list_items:
                    next_steps = block.list_items
                    break

    # Mapper les sections block-based vers CourseSection (format API)
    from app.api.schemas import CourseSection, CoursePitfall, Step, WorkedExample
    api_sections: list[CourseSection] = []
    pitfalls: list[CoursePitfall] = []
    _SKIPPED_SECTION_TYPES = {"common_pitfalls", "summary", "next_steps"}

    for i, section in enumerate(schema.sections):
        if section.type.value == "common_pitfalls":
            for block in section.blocks:
                if block.pitfall:
                    pitfalls.append(CoursePitfall(
                        description=block.pitfall.description,
                        why_it_happens=block.pitfall.why_it_happens,
                        how_to_avoid=block.pitfall.how_to_avoid,
                    ))
                elif block.text:
                    # Fallback pour un block TEXT non structuré (ancien format / omission du modèle)
                    pitfalls.append(CoursePitfall(description=block.text, why_it_happens="", how_to_avoid=""))
            continue
        if section.type.value in _SKIPPED_SECTION_TYPES:
            continue

        # Extraire les sous-sections (Quoi/Pourquoi/Comment)
        quoi_text = ""
        pourquoi_text = ""
        comment_text = ""
        worked_ex = None

        for sub in section.subsections:
            sub_text = " ".join(_block_to_text(b) for b in sub.blocks if _block_to_text(b))
            if "quoi" in sub.title.lower():
                quoi_text = sub_text
            elif "pourquoi" in sub.title.lower():
                pourquoi_text = sub_text
            elif "comment" in sub.title.lower():
                comment_text = sub_text
            for b in sub.blocks:
                if b.worked_example:
                    worked_ex = WorkedExample(
                        statement=b.worked_example.statement,
                        steps=[Step(id=str(idx + 1), content=s) for idx, s in enumerate(b.worked_example.steps)],
                        result=b.worked_example.result,
                    )

        # Si pas de sous-sections, extraire depuis les blocks directs
        if not section.subsections and section.blocks:
            direct_text = " ".join(_block_to_text(b) for b in section.blocks if _block_to_text(b))
            if section.type.value == "introduction":
                quoi_text = direct_text
            else:
                comment_text = direct_text

        if quoi_text or pourquoi_text or comment_text:
            api_sections.append(CourseSection(
                id=str(i),
                title=section.title,
                quoi=quoi_text,
                pourquoi=pourquoi_text,
                comment=comment_text,
                worked_example=WorkedExample(
                    statement=worked_ex.statement if worked_ex else "",
                    steps=worked_ex.steps if worked_ex else [],
                    result=worked_ex.result if worked_ex else "",
                ),
            ))

    # Construire answer depuis la première section avec Quoi/Pourquoi/Comment
    answer = None
    if api_sections:
        first = api_sections[0]
        answer = {
            "quoi": first.quoi,
            "pourquoi": first.pourquoi,
            "comment": first.comment,
            "worked_example": {
                "statement": first.worked_example.statement,
                "steps": first.worked_example.steps,
                "result": first.worked_example.result,
            },
            "key_points": schema.unconfirmed_points,
        }

    return CourseGenerationResponse(
        mode=schema.mode.value,
        format=schema.format.value,
        meta=CourseMeta(
            title=schema.meta.title,
            subject=schema.meta.subject,
            language=schema.meta.language,
            generated_at=schema.meta.generated_at.isoformat() if isinstance(schema.meta.generated_at, datetime) else str(schema.meta.generated_at),
        ),
        sources=all_sources,
        answer=answer,
        sections=api_sections or None,
        common_pitfalls=pitfalls or None,
        quiz=[
            {
                "question": q.question,
                "options": q.choices,
                "correct_option_index": q.correct_index,
                "explanation": q.explanation,
                "time_limit_seconds": 80 if q.requires_calculation else 45,
            }
            for q in schema.quiz
        ] or None,
        summary=summary or (schema.sections[0].title if schema.sections else ""),
        next_steps=next_steps or schema.unconfirmed_points,
    )


_MODE_TO_FORMAT: dict[str, str] = {
    "file_question": "focused_answer",
    "question_only": "focused_answer",
}


def _coerce_known_format(structured: dict[str, Any], mode: str) -> dict[str, Any]:
    """`format` est entièrement dérivé de `mode` — full_course n'existe pour
    l'instant que pour un futur mode fichier-seul non branché sur Gemini.

    Gemini a tendance à halluciner 'full_course' quand la question ressemble
    à 'explique le cours en complet', alors que le mode réel (file_question/
    question_only) impose focused_answer. On écrase donc la valeur côté code
    plutôt que de dépendre du modèle : ça élimine la classe d'erreur au lieu
    de la détecter, et évite de perdre un appel Gemini sur une regénération.
    """
    expected = _MODE_TO_FORMAT.get(mode)
    if expected is not None:
        structured["format"] = expected
    return structured


def _validate_and_map(structured: dict[str, Any], mode: str) -> CourseGenerationResponse:
    """Corrige le format déterministe, valide contre CourseGenerationSchema,
    puis mappe vers CourseGenerationResponse — sous un seul try/except.
    """
    try:
        structured = _coerce_known_format(structured, mode)
        gemini_result = CourseGenerationSchema.model_validate(structured)
        return _map_schema_to_response(gemini_result)
    except Exception as exc:
        raise GeminiInvalidResponseError(f"JSON structuré invalide: {exc}") from exc


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
        gemini_client: client Gemini.
        settings: configuration applicative.
        mode: "file_question" (Mode 2, retrieval RAG) ou "question_only" (Mode 3, recherche web).
        top_k: nombre de chunks à récupérer (défaut settings.course_top_k_default).
        filename: filtre optionnel sur un document déjà ingéré.

    Retour: CourseGenerationResponse validé.

    Fonctionnement:
        1. Retrieval des chunks pertinents dans le vector store local (sauf mode question_only).
        2. Si gemini_use_search_grounding=True : 2 appels (search_grounded + format_structured).
           Sinon : 1 seul appel (format_structured direct avec contexte RAG).

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

    is_question_only = mode == "question_only"

    # --- Mode 1 appel (pas de recherche web, quota minima) ---
    if not settings.gemini_use_search_grounding:
        if is_question_only:
            prompt = (
                f'mode="{mode}"\n'
                f"Question de l'utilisateur : {question}\n\n"
                f"Sources fichier disponibles : []\n"
                f"Sources web disponibles : []\n\n"
                f"Génère directement le JSON structuré selon le schéma fourni."
            )
        else:
            prompt = (
                f'mode="{mode}"\n'
                f"Question de l'utilisateur : {question}\n\n"
                f"Contexte extrait des documents fournis :\n{context_block}\n\n"
                f"Sources fichier disponibles : {file_sources}\n"
                f"Sources web disponibles : []\n\n"
                f"Génère directement le JSON structuré selon le schéma fourni."
            )
        structured = await gemini_client.format_structured(
            raw_answer=prompt,
            system_instruction=system_instruction,
            response_schema=CourseGenerationSchema,
        )
        return _validate_and_map(structured, mode)

    # --- Mode 2 appels (search grounding + reformatage) ---
    if is_question_only:
        prompt = f"Question de l'utilisateur : {question}"
    else:
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
        system_instruction=system_instruction,
        response_schema=CourseGenerationSchema,
    )
    return _validate_and_map(structured, mode)