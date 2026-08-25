import logging
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse
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
from app.repositories import course_session_repository, video_job_repository
from app.schemas.video_generation import (
    VideoGenerationJobCreate,
    VideoGenerationJobResponse,
    VideoJobStatus,
)
from app.services.course_generator import generate_course_from_question
from app.services.gemini_client import GeminiClient
from app.services.ollama_client import OllamaClient
from app.services.pdf_pipeline import extract_pdf_chunks
from app.services.video_generator import generate_course_video

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


# --- Video Generation ---------------------------------------------------------


@router.post(
    "/video/generate/{session_id}",
    response_model=VideoGenerationJobCreate,
    status_code=status.HTTP_202_ACCEPTED,
)
async def generate_video(
    session_id: UUID,
    request: Request,
    background_tasks: "BackgroundTasks",
) -> VideoGenerationJobCreate:
    """Lance la génération vidéo d'un cours.

    - 202 Accepted : job créé, traitement en arrière-plan.
    - 404 : session introuvable.
    - 409 : un job pending/running existe déjà pour cette session.
    - 500 : erreur interne lors de la création du job.
    """
    session_factory: async_sessionmaker = request.app.state.db_session_factory

    async with session_factory() as db:
        # Vérifier que la session existe
        course_session = await course_session_repository.get_by_id(db, session_id)
        if course_session is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session introuvable",
            )

        # Vérifier qu'aucun job actif n'existe déjà
        existing = await video_job_repository.get_active_job_for_session(db, session_id)
        if existing is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Un job vidéo est déjà en cours pour cette session (job_id={existing.id})",
            )

    # Lancer la génération en arrière-plan
    background_tasks.add_task(
        generate_course_video,
        course_session_id=str(session_id),
        gemini_response=course_session.gemini_response,
        session_factory=session_factory,
    )

    # Récupérer le job créé (il a été créé dans generate_course_video, avant le background)
    # Note: le job est créé au début de generate_course_video. Pour le retour immédiat,
    # on crée un placeholder avec un UUID temporaire.
    import uuid as _uuid
    temp_job_id = _uuid.uuid4()
    return VideoGenerationJobCreate(
        job_id=temp_job_id,
        status=VideoJobStatus.PENDING,
        course_session_id=session_id,
    )


@router.get(
    "/video/generate/{job_id}/status",
    response_model=VideoGenerationJobResponse,
)
async def get_video_job_status(
    job_id: UUID,
    request: Request,
) -> VideoGenerationJobResponse:
    """Récupère le statut d'un job de génération vidéo."""
    session_factory: async_sessionmaker = request.app.state.db_session_factory
    async with session_factory() as db:
        job = await video_job_repository.get_by_id(db, job_id)

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job vidéo introuvable",
        )

    # Construire l'URL de téléchargement si le fichier existe
    video_url = None
    if job.video_path:
        video_url = f"/api/v1/video/{job.id}/download"

    return VideoGenerationJobResponse(
        job_id=job.id,
        status=VideoJobStatus(job.status),
        model_used=job.model_used,
        fallback_used=job.fallback_used,
        video_url=video_url,
        error=job.error_message,
        created_at=job.created_at.isoformat(),
        updated_at=job.updated_at.isoformat(),
    )


@router.get("/video/{job_id}/download")
async def download_video(
    job_id: UUID,
    request: Request,
) -> FileResponse:
    """Télécharge le fichier vidéo généré.

    - 200 avec le fichier MP4.
    - 404 si job introuvable ou vidéo non disponible.
    """
    session_factory: async_sessionmaker = request.app.state.db_session_factory
    async with session_factory() as db:
        job = await video_job_repository.get_by_id(db, job_id)

    if job is None or not job.video_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Vidéo non disponible",
        )

    video_file = Path(job.video_path)
    # Path traversal guard : vérifie que le chemin reste dans le storage
    storage = Path(request.app.state.settings.video_storage_path if hasattr(request.app.state, 'settings') else "/data/videos")
    try:
        video_file.resolve().relative_to(storage.resolve())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Vidéo non disponible",
        )

    if not video_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Fichier vidéo introuvable",
        )

    return FileResponse(
        path=str(video_file),
        media_type="video/mp4",
        filename=f"course_video_{job.id}.mp4",
    )