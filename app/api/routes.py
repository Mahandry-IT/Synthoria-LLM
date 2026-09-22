import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.api.schemas import (
    COURSE_DEFAULT_QUESTION,
    ApiPlannedSection,
    ApiPretestItem,
    CourseFromPlanRequest,
    CourseGenerationRequest,
    CourseGenerationResponse,
    CourseHistoryDetail,
    CourseHistoryItem,
    CoursePlanDetail,
    CoursePlanMeta,
    CoursePlanRequest,
    CoursePlanResponse,
    DocumentQueryRequest,
    DocumentQueryResponse,
    FileListResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    MoreSectionsRequest,
    MoreSectionsResponse,
    PageParams,
    PaginatedResponse,
    PaginationMeta,
    PDFIngestMultiResponse,
    PDFIngestResponse,
    PendingPlanItem,
    RecallRequest,
    RecallResponse,
    RefineSectionRequest,
)
from app.core.config import Settings, get_settings
from app.core.rate_limit import SlidingWindowLimiter
from app.core.exceptions import (
    GeminiInvalidResponseError,
    GeminiQuotaExceededError,
    GeminiUnavailableError,
    OllamaModelNotFoundError,
    OllamaUnavailableError,
)
from app.repositories import course_plan_repository, course_session_repository
from app.schemas.course_generation import CoursePlanSchema, SectionType
from app.services.course_generator import _map_quiz_question, generate_course_from_question
from app.services.course_plan_generator import (
    generate_course_from_validated_plan,
    generate_course_plan,
    generate_more_sections,
    refine_planned_section,
)
from app.services.gemini_client import GeminiClient
from app.services.recall_evaluator import evaluate_recall
from app.services.ollama_client import OllamaClient
from app.services.pdf_pipeline import extract_pdf_chunks
from app.services.podcast.jobs import enqueue_podcast_job

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
    except (OllamaUnavailableError, OllamaModelNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except Exception:
        logger.exception("pdf_ingest_failed", extra={"pdf_filename": file.filename})
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


@contextmanager
def _gemini_http_errors() -> Iterator[None]:
    """Traduit les erreurs Gemini/Ollama de la génération de cours en erreurs HTTP."""
    try:
        yield
    except GeminiUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except GeminiQuotaExceededError as exc:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)) from exc
    except GeminiInvalidResponseError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    except (OllamaUnavailableError, OllamaModelNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc


def _resolve_question_and_mode(body: CourseGenerationRequest, settings: Settings) -> tuple[str, str]:
    """Question effective (défaut si vide, 413 si trop longue) et mode résolu.

    Mode : explicite > filename → file_question > question_only.
    """
    question = (body.question or "").strip() or COURSE_DEFAULT_QUESTION
    if len(question) > settings.course_question_max_length:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"question trop longue (max {settings.course_question_max_length} caractères)",
        )

    if body.mode:
        mode = body.mode
    elif body.filename:
        mode = "file_question"
    else:
        mode = "question_only"
    return question, mode


def _filenames_list(filename: str | list[str] | None) -> list[str]:
    if isinstance(filename, str):
        return [filename]
    return list(filename) if isinstance(filename, list) else []


async def _persist_course_session(
    request: Request,
    *,
    question: str,
    filenames: list[str],
    mode: str,
    response: CourseGenerationResponse,
) -> UUID | None:
    """Best-effort : persistance de la session en PostgreSQL (ne fait jamais échouer la requête).

    Retourne l'id de la session persistée, ou None si la persistance a échoué.
    """
    try:
        session_factory: async_sessionmaker = request.app.state.db_session_factory
        async with session_factory() as db:
            row = await course_session_repository.save(
                db, question=question, filenames=filenames, mode=mode, response=response,
            )
        return row.id if isinstance(row.id, UUID) else None
    except Exception:
        logger.error("course_session_persist_failed", exc_info=True)
        return None


async def _maybe_enqueue_podcast(
    request: Request,
    session_id: UUID | None,
    requested: bool | None,
    settings: Settings,
) -> UUID | None:
    """Met en file un podcast après la génération du cours, si demandé (ou activé par défaut).

    Best-effort : un échec est journalisé et renvoie None, jamais d'échec de la génération du cours.
    """
    wanted = settings.podcast_auto_generate if requested is None else requested
    if not (wanted and settings.podcast_enabled and session_id is not None):
        return None
    try:
        job, _ = await enqueue_podcast_job(request.app.state.db_session_factory, session_id, None, settings)
        return job.id
    except Exception:
        logger.error("podcast_auto_enqueue_failed", exc_info=True)
        return None


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
    question, resolved_mode = _resolve_question_and_mode(body, settings)
    vector_store = request.app.state.vector_store

    with _gemini_http_errors():
        course_response = await generate_course_from_question(
            question=question,
            vector_store=vector_store,
            gemini_client=gemini_client,
            settings=settings,
            mode=resolved_mode,
            top_k=body.top_k,
            filename=body.filename,
            full_document=body.full_document,
            db_session_factory=request.app.state.db_session_factory,
        )

    session_id = await _persist_course_session(
        request, question=question, filenames=_filenames_list(body.filename),
        mode=resolved_mode, response=course_response,
    )
    podcast_job_id = await _maybe_enqueue_podcast(request, session_id, body.generate_podcast, settings)
    return course_response.model_copy(update={"session_id": session_id, "podcast_job_id": podcast_job_id})


@router.post("/courses/plan", response_model=CoursePlanResponse)
async def create_course_plan(
    request: Request,
    body: CoursePlanRequest,
    gemini_client: GeminiClient = Depends(get_gemini_client),
    settings: Settings = Depends(get_settings),
) -> CoursePlanResponse:
    """Étape 1 : plan détaillé du cours (structure uniquement), à valider ou éditer.

    Le contexte de récupération est figé et persisté avec le plan : la
    génération complète (`/courses/generate/from-plan`) réutilise ce contexte.
    """
    question, resolved_mode = _resolve_question_and_mode(body, settings)
    vector_store = request.app.state.vector_store

    with _gemini_http_errors():
        plan, retrieval_context = await generate_course_plan(
            question=question,
            vector_store=vector_store,
            gemini_client=gemini_client,
            settings=settings,
            mode=resolved_mode,
            top_k=body.top_k,
            filename=body.filename,
            full_document=body.full_document,
        )

    expires_at = datetime.now(timezone.utc) + timedelta(minutes=settings.course_plan_ttl_minutes)
    try:
        session_factory: async_sessionmaker = request.app.state.db_session_factory
        async with session_factory() as db:
            row = await course_plan_repository.save(
                db,
                question=question,
                mode=resolved_mode,
                filenames=_filenames_list(body.filename),
                top_k=body.top_k,
                full_document=body.full_document,
                retrieval_context=retrieval_context,
                plan=plan.model_dump(mode="json"),
                expires_at=expires_at,
            )
    except Exception as exc:
        # Sans persistance, le plan_id renvoyé serait inutilisable : erreur explicite.
        logger.error("course_plan_persist_failed", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Persistance du plan indisponible",
        ) from exc

    return CoursePlanResponse(
        plan_id=row.id,
        expires_at=row.expires_at.isoformat(),
        mode=resolved_mode,
        meta=CoursePlanMeta(**plan.meta.model_dump()),
        sections=[ApiPlannedSection(**s.model_dump(mode="json")) for s in plan.planned_sections],
        pretest=_api_pretest(plan),
        coverage_notes=plan.coverage_notes,
    )


def _api_pretest(plan: CoursePlanSchema) -> list[ApiPretestItem]:
    """Pré-test du plan côté API : 1 question par section de développement existante (le reste est ignoré)."""
    titles = {s.title.strip().casefold() for s in plan.planned_sections if s.type is SectionType.DEVELOPMENT}
    seen: set[str] = set()
    items: list[ApiPretestItem] = []
    for item in plan.pretest:
        key = item.section_title.strip().casefold()
        if key in titles and key not in seen:
            seen.add(key)
            items.append(ApiPretestItem(section_title=item.section_title, question=_map_quiz_question(item.question)))
    return items


async def _get_active_plan(request: Request, plan_id: UUID):
    """Plan persisté non expiré, sinon 404 (inconnu) / 410 (expiré)."""
    session_factory: async_sessionmaker = request.app.state.db_session_factory
    async with session_factory() as db:
        plan_row = await course_plan_repository.get_by_id(db, plan_id)

    if plan_row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Plan introuvable")
    if plan_row.expires_at <= datetime.now(timezone.utc):
        raise HTTPException(status_code=status.HTTP_410_GONE, detail="Plan expiré, régénérez-le")
    return plan_row


@router.post("/courses/plan/refine-section", response_model=ApiPlannedSection)
async def refine_plan_section(
    request: Request,
    body: RefineSectionRequest,
    gemini_client: GeminiClient = Depends(get_gemini_client),
    settings: Settings = Depends(get_settings),
) -> ApiPlannedSection:
    """Complète une section de plan incomplète (bouton « étoiles » d'une section).

    Reçoit la section actuelle et, facultativement, ce que l'utilisateur veut y
    ajouter ; sans précision, le modèle détermine seul ce qui manque.
    404 si `plan_id` inconnu, 410 si le plan a expiré.
    """
    plan_row = await _get_active_plan(request, body.plan_id)

    with _gemini_http_errors():
        return await refine_planned_section(
            plan_row=plan_row,
            section=body.section,
            outline=body.sections,
            instructions=body.instructions,
            vector_store=request.app.state.vector_store,
            gemini_client=gemini_client,
            settings=settings,
        )


def _limit_more_sections(request: Request, settings: Settings = Depends(get_settings)) -> None:
    limiter = getattr(request.app.state, "more_sections_rate_limiter", None)
    if limiter is None:
        limiter = request.app.state.more_sections_rate_limiter = SlidingWindowLimiter()
    limiter.check(
        request.client.host if request.client else "unknown",
        settings.more_sections_rate_limit_per_minute,
        "Trop de demandes de nouvelles sections, réessayez dans une minute",
    )


@router.post(
    "/courses/plan/more-sections",
    response_model=MoreSectionsResponse,
    dependencies=[Depends(_limit_more_sections)],
)
async def add_more_plan_sections(
    request: Request,
    body: MoreSectionsRequest,
    gemini_client: GeminiClient = Depends(get_gemini_client),
) -> MoreSectionsResponse:
    """Crée de nouvelles sections de développement à partir de « Pour aller plus loin ».

    La réponse contient aussi `next_steps` : la section « Pour aller plus loin » actualisée avec de
    nouvelles pistes, que le client doit substituer à l'ancienne.

    404 si `plan_id` inconnu, 410 si le plan a expiré.
    """
    plan_row = await _get_active_plan(request, body.plan_id)

    with _gemini_http_errors():
        result = await generate_more_sections(
            plan_row=plan_row, current_sections=body.sections, gemini_client=gemini_client
        )
    return MoreSectionsResponse(sections=result.sections, next_steps=result.next_steps)


@router.get("/courses/plans", response_model=PaginatedResponse[PendingPlanItem])
async def list_pending_course_plans(
    request: Request,
    pagination: PageParams = Depends(),
) -> PaginatedResponse[PendingPlanItem]:
    """Plans en cours : proposés, non expirés et pas encore transformés en cours."""
    session_factory: async_sessionmaker = request.app.state.db_session_factory
    async with session_factory() as db:
        rows, total = await course_plan_repository.list_pending(
            db, page=pagination.page, limit=pagination.limit, now=datetime.now(timezone.utc)
        )

    total_pages = max(1, (total + pagination.limit - 1) // pagination.limit)
    items = [
        PendingPlanItem(
            plan_id=row.id,
            question=row.question,
            title=row.plan["meta"]["title"],
            subject=row.plan["meta"].get("subject", ""),
            sections_count=len(row.plan["planned_sections"]),
            created_at=row.created_at.isoformat(),
            expires_at=row.expires_at.isoformat(),
        )
        for row in rows
    ]
    return PaginatedResponse(
        data=items,
        meta=PaginationMeta(
            page=min(pagination.page, total_pages) if total > 0 else 1,
            limit=pagination.limit,
            total=total,
            totalPages=total_pages,
        ),
    )


@router.get("/courses/plans/{plan_id}", response_model=CoursePlanDetail)
async def get_course_plan(request: Request, plan_id: UUID) -> CoursePlanDetail:
    """Plan proposé, relu tel quel (reprise depuis le dashboard). 404 inconnu, 410 expiré."""
    plan_row = await _get_active_plan(request, plan_id)
    plan = CoursePlanSchema.model_validate(plan_row.plan)
    return CoursePlanDetail(
        plan_id=plan_row.id,
        expires_at=plan_row.expires_at.isoformat(),
        mode=plan_row.mode,
        meta=CoursePlanMeta(**plan.meta.model_dump()),
        sections=[ApiPlannedSection(**s.model_dump(mode="json")) for s in plan.planned_sections],
        pretest=_api_pretest(plan),
        coverage_notes=plan.coverage_notes,
        question=plan_row.question,
        filenames=list(plan_row.filenames),
    )


@router.post("/courses/generate/from-plan", response_model=CourseGenerationResponse)
async def generate_course_from_plan(
    request: Request,
    body: CourseFromPlanRequest,
    gemini_client: GeminiClient = Depends(get_gemini_client),
    settings: Settings = Depends(get_settings),
) -> CourseGenerationResponse:
    """Étape 2 : cours complet à partir du plan validé (éventuellement édité).

    404 si `plan_id` inconnu, 410 si le plan a expiré. La structure envoyée
    par le client est revalidée (Pydantic) ; question, mode et contexte
    viennent exclusivement du plan persisté.
    """
    session_factory: async_sessionmaker = request.app.state.db_session_factory
    async with session_factory() as db:
        plan_row = await course_plan_repository.get_by_id(db, body.plan_id)

    if plan_row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Plan introuvable")
    if plan_row.expires_at <= datetime.now(timezone.utc):
        raise HTTPException(status_code=status.HTTP_410_GONE, detail="Plan expiré, régénérez-le")

    with _gemini_http_errors():
        course_response = await generate_course_from_validated_plan(
            plan_row=plan_row,
            edited_sections=body.sections,
            gemini_client=gemini_client,
            settings=settings,
            db_session_factory=session_factory,
        )

    session_id = await _persist_course_session(
        request, question=plan_row.question, filenames=list(plan_row.filenames),
        mode=plan_row.mode, response=course_response,
    )
    podcast_job_id = await _maybe_enqueue_podcast(request, session_id, body.generate_podcast, settings)
    course_response = course_response.model_copy(update={"session_id": session_id, "podcast_job_id": podcast_job_id})
    try:
        async with session_factory() as db:
            await course_plan_repository.mark_generated(db, plan_row.id)
    except Exception:
        logger.error("course_plan_mark_generated_failed", exc_info=True)

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


def _limit_recall(request: Request, settings: Settings = Depends(get_settings)) -> None:
    limiter = getattr(request.app.state, "recall_rate_limiter", None)
    if limiter is None:
        limiter = request.app.state.recall_rate_limiter = SlidingWindowLimiter()
    limiter.check(
        request.client.host if request.client else "unknown",
        settings.recall_rate_limit_per_minute,
        "Trop d'évaluations, réessayez dans une minute",
    )


@router.post(
    "/courses/{session_id}/sections/{section_id}/recall",
    response_model=RecallResponse,
    dependencies=[Depends(_limit_recall)],
)
async def evaluate_section_recall(
    session_id: UUID,
    section_id: str,
    body: RecallRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
    gemini_client: GeminiClient = Depends(get_gemini_client),
) -> RecallResponse:
    """Évalue la reformulation de l'apprenant pour une section d'une session persistée.

    La section (consigne + points attendus) est lue en base, jamais fournie par le client.
    404 si la session, la section ou sa consigne `recall_prompt` n'existe pas ; 422 si la réponse est
    vide ou dépasse `recall_answer_max_length` ; 429 au-delà de `recall_rate_limit_per_minute`.
    """
    if len(body.answer) > settings.recall_answer_max_length:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Réponse trop longue")

    session_factory: async_sessionmaker = request.app.state.db_session_factory
    async with session_factory() as db:
        row = await course_session_repository.get_by_id(db, session_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session introuvable")

    section = next(
        (s for s in (row.gemini_response.get("sections") or []) if str(s.get("id")) == section_id), None
    )
    if section is None or not section.get("recall_prompt"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Section introuvable")

    with _gemini_http_errors():
        evaluation = await evaluate_recall(section, body.answer, gemini_client)
    return RecallResponse(
        verdict=evaluation.verdict.value, feedback=evaluation.feedback, missing_points=evaluation.missing_points
    )
