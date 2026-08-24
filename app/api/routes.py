import logging
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.api.schemas import (
    COURSE_DEFAULT_QUESTION,
    CourseGenerationRequest,
    CourseGenerationResponse,
    CourseHistoryDetail,
    CourseHistoryItem,
    DocumentQueryRequest,
    DocumentQueryResponse,
    FileListResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    PageParams,
    PaginatedResponse,
    PaginationMeta,
    PDFIngestMultiResponse,
    PDFIngestResponse,
)
from app.core.config import Settings, get_settings
from app.core.exceptions import (
    GeminiInvalidResponseError,
    GeminiQuotaExceededError,
    GeminiUnavailableError,
    OllamaModelNotFoundError,
    OllamaUnavailableError,
)
from app.repositories import course_session_repository
from app.services.course_generator import generate_course_from_question
from app.services.gemini_client import GeminiClient
from app.services.ollama_client import OllamaClient
from app.services.pdf_pipeline import extract_pdf_chunks

logger = logging.getLogger(__name__)

router = APIRouter()


def get_ollama_client(request: Request) -> OllamaClient:
    return request.app.state.ollama_client


def get_gemini_client(request: Request) -> GeminiClient:
    return request.app.state.gemini_client


def get_db_session_factory(request: Request) -> async_sessionmaker:
    return request.app.state.db_session_factory


@router.get("/health", response_model=HealthResponse)
async def health(client: OllamaClient = Depends(get_ollama_client)) -> HealthResponse:
    reachable = await client.is_reachable()
    return HealthResponse(status="ok", ollama_reachable=reachable)


@router.post("/generate", response_model=GenerateResponse)
async def generate(
    body: GenerateRequest,
    client: OllamaClient = Depends(get_ollama_client),
    settings: Settings = Depends(get_settings),
) -> GenerateResponse:
    if len(body.prompt) > settings.request_max_prompt_length:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"prompt trop long (max {settings.request_max_prompt_length} caractères)",
        )

    model = body.model or settings.ollama_default_model

    try:
        result = await client.generate(prompt=body.prompt, model=model, stream=False)
    except OllamaModelNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except OllamaUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc

    return GenerateResponse(
        model=result.get("model", model),
        response=result.get("response", ""),
        done=result.get("done", True),
    )


async def _ingest_single_pdf(
    request: Request,
    file: UploadFile,
    settings: Settings,
) -> PDFIngestResponse:
    """Ingestion d'un seul fichier PDF — logique partagée single/multi."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return PDFIngestResponse(
            status="error",
            filename=file.filename or "unknown",
            chunks_added=0,
            documents_added=0,
        )

    # Détection de doublon : vérifier si le fichier existe déjà dans le store
    vector_store = request.app.state.vector_store
    if vector_store.has_file(file.filename):
        return PDFIngestResponse(
            status="failed",
            filename=file.filename,
            chunks_added=0,
            documents_added=0,
            message="File already uploaded",
        )

    try:
        content = await file.read()
        chunks = extract_pdf_chunks(content, file.filename, settings)
        if not chunks:
            return PDFIngestResponse(
                status="error",
                filename=file.filename,
                chunks_added=0,
                documents_added=0,
            )

        added = await vector_store.add_chunks(chunks)
        return PDFIngestResponse(
            status="ok",
            filename=file.filename,
            chunks_added=added,
            documents_added=len(chunks),
        )
    except Exception as exc:
        return PDFIngestResponse(
            status="error",
            filename=file.filename or "unknown",
            chunks_added=0,
            documents_added=0,
        )


@router.post("/pdf/ingest")
async def ingest_pdf(
    request: Request,
    files: list[UploadFile] = File(...),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Ingestion de un ou plusieurs fichiers PDF dans le vector store.

    Retourne toujours PDFIngestMultiResponse (compatible 1 ou N fichiers).
    """
    results: list[PDFIngestResponse] = []
    for file in files:
        results.append(await _ingest_single_pdf(request, file, settings))

    if not results:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Aucun fichier PDF fourni",
        )

    # Rétrocompatibilité : si un seul fichier envoyé et succès, format simple
    if len(files) == 1 and results[0].status == "ok":
        return results[0].model_dump()

    total_chunks = sum(r.chunks_added for r in results)
    total_docs = sum(r.documents_added for r in results)
    return PDFIngestMultiResponse(
        status="ok",
        files=results,
        total_chunks=total_chunks,
        total_documents=total_docs,
    ).model_dump()


@router.post("/pdf/search", response_model=DocumentQueryResponse)
async def search_pdf(
    request: Request,
    body: DocumentQueryRequest,
) -> DocumentQueryResponse:
    """Recherche vectorielle avec filtre optionnel sur un ou plusieurs fichiers."""
    vector_store = request.app.state.vector_store
    # Le filtre filename est appliqué côté vector_store, avant le classement
    # top_k : garantit que chaque fichier ciblé est effectivement représenté,
    # au lieu de dépendre d'un sur-échantillonnage suivi d'un post-filtrage.
    results = await vector_store.search(
        body.query, top_k=body.top_k, filename_filter=body.filename
    )

    return DocumentQueryResponse(query=body.query, results=results)


@router.get("/pdf/files", response_model=FileListResponse)
async def list_files(
    request: Request,
    pagination: PageParams = Depends(),
) -> FileListResponse:
    """Liste les fichiers PDF stockés dans le vector store (paginé)."""
    vector_store = request.app.state.vector_store
    all_files = vector_store.list_files()

    total = len(all_files)
    total_pages = max(1, (total + pagination.limit - 1) // pagination.limit)
    page = min(pagination.page, total_pages) if total > 0 else 1
    offset = (page - 1) * pagination.limit
    paginated = all_files[offset : offset + pagination.limit]

    return FileListResponse(
        data=paginated,
        meta=PaginationMeta(
            page=page,
            limit=pagination.limit,
            total=total,
            totalPages=total_pages,
        ),
    )


@router.post("/courses/generate", response_model=CourseGenerationResponse)
async def generate_course(
    request: Request,
    body: CourseGenerationRequest,
    gemini_client: GeminiClient = Depends(get_gemini_client),
    settings: Settings = Depends(get_settings),
) -> CourseGenerationResponse:
    """Mode 2 (fichier + question) ou Mode 3 (question seule + recherche web).

    `full_document=True` bascule le retrieval en mode exhaustif : tous les
    chunks du/des fichier(s) filtré(s) sont utilisés au lieu du top-k par
    similarité, au prix d'un contexte plus volumineux envoyé à Gemini.
    """
    question = (body.question or "").strip() or COURSE_DEFAULT_QUESTION
    if len(question) > settings.course_question_max_length:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"question trop longue (max {settings.course_question_max_length} caractères)",
        )

    # Auto-détection du mode : explicite > filename → file_question > question_only
    if body.mode:
        resolved_mode = body.mode
    elif body.filename:
        resolved_mode = "file_question"
    else:
        resolved_mode = "question_only"

    vector_store = request.app.state.vector_store

    try:
        course_response = await generate_course_from_question(
            question=question,
            vector_store=vector_store,
            gemini_client=gemini_client,
            settings=settings,
            mode=resolved_mode,
            top_k=body.top_k,
            filename=body.filename,
            full_document=body.full_document,
        )
    except GeminiUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except GeminiQuotaExceededError as exc:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)) from exc
    except GeminiInvalidResponseError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    except (OllamaUnavailableError, OllamaModelNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    # Best-effort : persistance de la session en PostgreSQL
    filenames_list = (
        [body.filename] if isinstance(body.filename, str)
        else body.filename if isinstance(body.filename, list)
        else []
    )
    try:
        session_factory: async_sessionmaker = request.app.state.db_session_factory
        async with session_factory() as db:
            await course_session_repository.save(
                db,
                question=question,
                filenames=filenames_list,
                mode=resolved_mode,
                response=course_response,
            )
    except Exception:
        logger.error("course_session_persist_failed", exc_info=True)

    return course_response


@router.get("/courses/history", response_model=PaginatedResponse[CourseHistoryItem])
async def list_course_history(
    request: Request,
    pagination: PageParams = Depends(),
) -> PaginatedResponse[CourseHistoryItem]:
    """Historique paginé des sessions de génération de cours."""
    session_factory: async_sessionmaker = request.app.state.db_session_factory
    async with session_factory() as db:
        rows, total = await course_session_repository.list_paginated(
            db, page=pagination.page, limit=pagination.limit
        )

    total_pages = max(1, (total + pagination.limit - 1) // pagination.limit)
    page = min(pagination.page, total_pages) if total > 0 else 1

    items = [
        CourseHistoryItem(
            id=row.id,
            created_at=row.created_at.isoformat(),
            question=row.question,
            filenames=row.filenames,
            mode=row.mode,
        )
        for row in rows
    ]

    return PaginatedResponse(
        data=items,
        meta=PaginationMeta(
            page=page,
            limit=pagination.limit,
            total=total,
            totalPages=total_pages,
        ),
    )


@router.get("/courses/history/{session_id}", response_model=CourseHistoryDetail)
async def get_course_history(
    session_id: UUID,
    request: Request,
) -> CourseHistoryDetail:
    """Détail d'une session de cours (404 si introuvable)."""
    session_factory: async_sessionmaker = request.app.state.db_session_factory
    async with session_factory() as db:
        row = await course_session_repository.get_by_id(db, session_id)

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session introuvable",
        )

    return CourseHistoryDetail(
        id=row.id,
        created_at=row.created_at.isoformat(),
        question=row.question,
        filenames=row.filenames,
        mode=row.mode,
        gemini_response=row.gemini_response,
    )