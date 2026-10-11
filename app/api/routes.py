import logging
import math
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.api.schemas import (
    COURSE_DEFAULT_QUESTION,
    AddCourseSectionsRequest,
    AddCourseSectionsResponse,
    ApiPlannedSection,
    ApiPretestItem,
    ChallengeRequest,
    ChallengeResponse,
    CourseFromPlanRequest,
    CourseGenerationRequest,
    CourseGenerationResponse,
    CourseHistoryDetail,
    CourseHistoryItem,
    CoursePlanDetail,
    CoursePlanMeta,
    CoursePlanRequest,
    CoursePlanResponse,
    CourseSection,
    DocumentQueryRequest,
    DocumentQueryResponse,
    FileFolderResult,
    FileInfo,
    FileListResponse,
    FolderMoveResult,
    FolderSummary,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    MoreSectionsRequest,
    MoreSectionsResponse,
    MoveCourseFolderRequest,
    MoveFileFolderRequest,
    PageParams,
    PaginatedResponse,
    PaginationMeta,
    PDFIngestMultiResponse,
    PDFIngestResponse,
    PendingPlanItem,
    RecallRequest,
    RecallResponse,
    RefineSectionRequest,
    SectionNoteRequest,
    SectionNoteResponse,
    SubfolderSummary,
    VideoNoteRequest,
    VideoNoteResponse,
)
from app.core.config import Settings, get_settings
from app.core.errors import ApiError, ErrorCode, exception_debug
from app.core.rate_limit import SlidingWindowLimiter
from app.core.exceptions import (
    GeminiDailyQuotaExceededError,
    GeminiInvalidResponseError,
    GeminiQuotaExceededError,
    GeminiUnavailableError,
    OllamaModelNotFoundError,
    OllamaUnavailableError,
)
from app.db.models import DEFAULT_COURSE_FOLDER, DEFAULT_COURSE_SUBFOLDER
from app.repositories import (
    course_plan_repository,
    course_section_note_repository,
    course_session_repository,
    course_video_note_repository,
    ingested_file_repository,
)
from app.schemas.course_generation import CoursePlanSchema, SectionType
from app.services.challenge_evaluator import evaluate_challenge
from app.services.course_depth import depth_of_plan
from app.services.course_generator import (
    _map_quiz_question,
    _map_sections_to_course_sections,
    generate_course_from_question,
    is_incomplete_section_dict,
)
from app.services.course_plan_generator import (
    generate_course_from_validated_plan,
    generate_course_plan,
    generate_more_sections,
    refine_planned_section,
)
from app.services.course_section_adder import add_course_sections
from app.services.gemini_client import GeminiClient
from app.services.media.visual_resolver import resolve_visuals_in_sections
from app.services.recall_evaluator import evaluate_recall
from app.services.section_regenerator import regenerate_section
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
async def health(
    client: OllamaClient = Depends(get_ollama_client),
    gemini_client: GeminiClient = Depends(get_gemini_client),
) -> HealthResponse:
    reachable = await client.is_reachable()
    gemini_health = await gemini_client.health_snapshot()
    return HealthResponse(status="ok", ollama_reachable=reachable, gemini=gemini_health)


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

    started_at = time.perf_counter()
    try:
        content = await file.read()
        log_extra = {"pdf_filename": file.filename, "bytes": len(content)}
        chunks = await extract_pdf_chunks(content, file.filename, settings, request.app.state.gemini_client)
        if not chunks:
            logger.warning("pdf_ingest_no_chunks", extra=log_extra)
            return PDFIngestResponse(
                status="error",
                filename=file.filename,
                chunks_added=0,
                documents_added=0,
            )

        added = await vector_store.add_chunks(chunks)
        logger.info(
            "pdf_ingest_succeeded",
            extra={**log_extra, "chunks": len(chunks), "elapsed_seconds": round(time.perf_counter() - started_at, 1)},
        )
        return PDFIngestResponse(
            status="ok",
            filename=file.filename,
            chunks_added=added,
            documents_added=len(chunks),
        )
    except (OllamaUnavailableError, OllamaModelNotFoundError) as exc:
        logger.warning(
            "pdf_ingest_ollama_unavailable",
            extra={"pdf_filename": file.filename, "elapsed_seconds": round(time.perf_counter() - started_at, 1), "error": str(exc)},
        )
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except Exception:
        logger.exception(
            "pdf_ingest_failed",
            extra={"pdf_filename": file.filename, "elapsed_seconds": round(time.perf_counter() - started_at, 1)},
        )
        return PDFIngestResponse(
            status="error",
            filename=file.filename or "unknown",
            chunks_added=0,
            documents_added=0,
        )


def _ingest_folder_target(
    folder: str | None = Form(None, description="Dossier cible (défaut : dossier par défaut)"),
    subfolder: str | None = Form(None, description="Sous-dossier cible (défaut : sous-dossier par défaut)"),
) -> MoveFileFolderRequest | None:
    """Dossier cible optionnel de l'ingestion, validé comme un déplacement (422 sinon).

    None si aucun des deux champs n'est fourni : les fichiers restent dans le dossier par défaut
    sans qu'aucune ligne de rangement ne soit créée.
    """
    if folder is None and subfolder is None:
        return None
    try:
        return MoveFileFolderRequest(folder=folder if folder is not None else DEFAULT_COURSE_FOLDER, subfolder=subfolder)
    except ValidationError as exc:
        raise RequestValidationError(exc.errors(include_url=False)) from exc


async def _place_ingested_files(
    request: Request, filenames: list[str], target: MoveFileFolderRequest
) -> None:
    """Range les fichiers fraîchement ingérés dans le dossier cible.

    Best-effort : les fichiers sont déjà dans le vector store, un échec ici ne doit pas faire
    croire à un échec d'ingestion (le client réessaierait et obtiendrait « File already uploaded »).
    Les fichiers restent alors dans le dossier par défaut et peuvent être déplacés ensuite.
    """
    session_factory: async_sessionmaker = request.app.state.db_session_factory
    try:
        async with session_factory() as db:
            await ingested_file_repository.move_many(
                db, filenames, folder=target.folder, subfolder=target.subfolder,
            )
    except Exception:
        logger.exception("pdf_ingest_folder_placement_failed", extra={"files": len(filenames)})


@router.post("/pdf/ingest")
async def ingest_pdf(
    request: Request,
    files: list[UploadFile] = File(...),
    target: MoveFileFolderRequest | None = Depends(_ingest_folder_target),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Ingestion de un ou plusieurs fichiers PDF dans le vector store, rangés directement dans le
    dossier/sous-dossier `folder`/`subfolder` s'ils sont fournis (multipart, optionnels).

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

    ingested = [r.filename for r in results if r.status == "ok"]
    if target is not None and ingested:
        await _place_ingested_files(request, ingested, target)

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
    """Liste les fichiers PDF stockés dans le vector store (paginé), avec leur dossier/sous-dossier
    (dossier par défaut pour un fichier jamais déplacé)."""
    vector_store = request.app.state.vector_store
    all_files = vector_store.list_files()

    total = len(all_files)
    total_pages = max(1, (total + pagination.limit - 1) // pagination.limit)
    page = min(pagination.page, total_pages) if total > 0 else 1
    offset = (page - 1) * pagination.limit
    paginated = all_files[offset : offset + pagination.limit]

    placements: dict[str, tuple[str, str]] = {}
    if paginated:
        session_factory: async_sessionmaker = request.app.state.db_session_factory
        async with session_factory() as db:
            placements = await ingested_file_repository.get_placements(db, (f["filename"] for f in paginated))
    default_placement = (DEFAULT_COURSE_FOLDER, DEFAULT_COURSE_SUBFOLDER)

    return FileListResponse(
        data=[
            FileInfo(
                id=f["id"],
                filename=f["filename"],
                folder=placements.get(f["filename"], default_placement)[0],
                subfolder=placements.get(f["filename"], default_placement)[1],
            )
            for f in paginated
        ],
        meta=PaginationMeta(
            page=page,
            limit=pagination.limit,
            total=total,
            totalPages=total_pages,
        ),
    )


@router.delete("/pdf/files/{filename}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_file(filename: str, request: Request) -> None:
    """Supprime un fichier PDF, tous ses chunks du vector store et sa ligne de rangement. 404 si
    inconnu.

    La ligne SQL est supprimée d'abord : si la base échoue, rien n'est perdu et l'appel peut être
    rejoué ; l'ordre inverse laisserait une ligne orpheline qui rangerait un futur fichier de même
    nom dans l'ancien dossier. Ne vérifie pas si ce fichier est encore référencé par un plan de
    cours en attente — voir `NumpyVectorStore.remove_file`.
    """
    session_factory: async_sessionmaker = request.app.state.db_session_factory
    async with session_factory() as db:
        await ingested_file_repository.delete(db, filename)
    vector_store = request.app.state.vector_store
    removed = vector_store.remove_file(filename)
    if not removed:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Fichier introuvable")


@router.put("/pdf/files/{filename}/folder", response_model=FileFolderResult)
async def move_file_folder(
    filename: str, body: MoveFileFolderRequest, request: Request,
) -> FileFolderResult:
    """Déplace un fichier ingéré vers un dossier/sous-dossier, créés implicitement (casse
    existante réutilisée). Idempotent. 404 si le fichier est inconnu du vector store."""
    if not request.app.state.vector_store.has_file(filename):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Fichier introuvable")
    session_factory: async_sessionmaker = request.app.state.db_session_factory
    async with session_factory() as db:
        folder, subfolder = await ingested_file_repository.move(
            db, filename, folder=body.folder, subfolder=body.subfolder,
        )
    return FileFolderResult(filename=filename, folder=folder, subfolder=subfolder)


@router.delete("/pdf/folders/{folder_name}", response_model=FolderMoveResult)
async def delete_file_folder(folder_name: str, request: Request) -> FolderMoveResult:
    """Supprime un dossier de fichiers : ses fichiers rejoignent le dossier par défaut (aucun
    fichier n'est supprimé). 400 si c'est le dossier par défaut."""
    _reject_default_folder(folder_name)
    session_factory: async_sessionmaker = request.app.state.db_session_factory
    async with session_factory() as db:
        moved = await ingested_file_repository.delete_folder(db, folder_name)
    return FolderMoveResult(moved=moved)


@router.delete("/pdf/folders/{folder_name}/subfolders/{subfolder_name}", response_model=FolderMoveResult)
async def delete_file_subfolder(
    folder_name: str, subfolder_name: str, request: Request,
) -> FolderMoveResult:
    """Supprime un sous-dossier de fichiers : ses fichiers rejoignent le sous-dossier par défaut
    du même dossier. 400 si c'est le sous-dossier par défaut."""
    _reject_default_subfolder(subfolder_name)
    session_factory: async_sessionmaker = request.app.state.db_session_factory
    async with session_factory() as db:
        moved = await ingested_file_repository.delete_subfolder(db, folder_name, subfolder_name)
    return FolderMoveResult(moved=moved)


# Repli si aucun `retry_at` exploitable (quota minute, classification "unknown" — voir
# `_classify_quota_error`) : une indication courte plutôt qu'aucune, sans prétendre à la précision
# d'un vrai quota jour.
_DEFAULT_QUOTA_RETRY_AFTER_SECONDS = 60


_GEMINI_QUOTA_DETAIL = "Quota Gemini atteint."
_GEMINI_DAILY_QUOTA_DETAIL = "Quota Gemini journalier atteint."
_GEMINI_UNAVAILABLE_DETAIL = "Le service Gemini est momentanément indisponible."
_GEMINI_INVALID_DETAIL = "La réponse de Gemini est inexploitable, réessayez."
_OLLAMA_UNAVAILABLE_DETAIL = "Le service d'embeddings local (Ollama) est indisponible."


@contextmanager
def _gemini_http_errors() -> Iterator[None]:
    """Traduit les erreurs Gemini/Ollama de la génération de cours en erreurs HTTP.

    Le `detail` est un message utilisateur fixe : le message brut (qui peut citer la réponse de
    Google) ne part qu'en `debug`, exposé seulement en développement.
    """
    try:
        yield
    except GeminiDailyQuotaExceededError as exc:
        retry_after = _DEFAULT_QUOTA_RETRY_AFTER_SECONDS
        if exc.retry_at is not None:
            retry_after = max(1, round((exc.retry_at - datetime.now(timezone.utc)).total_seconds()))
        raise ApiError(
            status.HTTP_429_TOO_MANY_REQUESTS, _GEMINI_DAILY_QUOTA_DETAIL, ErrorCode.GEMINI_QUOTA,
            headers={"Retry-After": str(retry_after)}, debug=exception_debug(exc),
        ) from exc
    except GeminiUnavailableError as exc:
        raise ApiError(
            status.HTTP_503_SERVICE_UNAVAILABLE, _GEMINI_UNAVAILABLE_DETAIL, ErrorCode.GEMINI_UNAVAILABLE,
            debug=exception_debug(exc),
        ) from exc
    except GeminiQuotaExceededError as exc:
        retry_after = _DEFAULT_QUOTA_RETRY_AFTER_SECONDS
        if exc.retry_after_seconds:
            retry_after = max(1, math.ceil(exc.retry_after_seconds))
        raise ApiError(
            status.HTTP_429_TOO_MANY_REQUESTS, _GEMINI_QUOTA_DETAIL, ErrorCode.GEMINI_QUOTA,
            headers={"Retry-After": str(retry_after)}, debug=exception_debug(exc),
        ) from exc
    except GeminiInvalidResponseError as exc:
        raise ApiError(
            status.HTTP_502_BAD_GATEWAY, _GEMINI_INVALID_DETAIL, ErrorCode.GEMINI_INVALID_RESPONSE,
            debug=exception_debug(exc),
        ) from exc
    except (OllamaUnavailableError, OllamaModelNotFoundError) as exc:
        raise ApiError(
            status.HTTP_503_SERVICE_UNAVAILABLE, _OLLAMA_UNAVAILABLE_DETAIL, ErrorCode.OLLAMA_UNAVAILABLE,
            debug=exception_debug(exc),
        ) from exc


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

    with gemini_client.track_calls(), _gemini_http_errors():
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
            depth=body.depth,
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
    Le mode (`depth`) est persisté avec le plan et pilote toute la génération du cours.
    """
    question, resolved_mode = _resolve_question_and_mode(body, settings)
    vector_store = request.app.state.vector_store

    with gemini_client.track_calls(), _gemini_http_errors():
        plan, retrieval_context = await generate_course_plan(
            question=question,
            vector_store=vector_store,
            gemini_client=gemini_client,
            settings=settings,
            mode=resolved_mode,
            top_k=body.top_k,
            filename=body.filename,
            full_document=body.full_document,
            depth=body.depth,
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
                depth=body.depth,
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
        depth=body.depth,
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
        depth=depth_of_plan(plan_row),
        meta=CoursePlanMeta(**plan.meta.model_dump()),
        sections=[ApiPlannedSection(**s.model_dump(mode="json")) for s in plan.planned_sections],
        pretest=_api_pretest(plan),
        coverage_notes=plan.coverage_notes,
        question=plan_row.question,
        filenames=list(plan_row.filenames),
    )


@router.delete("/courses/plans/{plan_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_course_plan(request: Request, plan_id: UUID) -> None:
    """Supprime un plan proposé. 404 si inconnu (un plan expiré peut être supprimé)."""
    session_factory: async_sessionmaker = request.app.state.db_session_factory
    async with session_factory() as db:
        deleted = await course_plan_repository.delete(db, plan_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Plan introuvable")


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

    with gemini_client.track_calls(), _gemini_http_errors():
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
    folder: str | None = Query(None, description="Filtrer par dossier"),
    subfolder: str | None = Query(None, description="Filtrer par sous-dossier"),
) -> PaginatedResponse[CourseHistoryItem]:
    """Historique paginé des sessions de génération de cours, filtrable par dossier/sous-dossier."""
    session_factory: async_sessionmaker = request.app.state.db_session_factory
    async with session_factory() as db:
        rows, total = await course_session_repository.list_paginated(
            db, page=pagination.page, limit=pagination.limit, folder=folder, subfolder=subfolder,
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
            folder=row.folder,
            subfolder=row.subfolder,
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
    """Détail d'une session de cours (404 si introuvable). Les notes de l'apprenant (tables séparées,
    jamais générées) sont fusionnées dans `gemini_response.sections[].note` et
    `gemini_response.videos[].note`, et `incomplete` est recalculé depuis le contenu (jamais la
    seule valeur stockée, qui peut dater d'avant ce champ)."""
    session_factory: async_sessionmaker = request.app.state.db_session_factory
    async with session_factory() as db:
        row = await course_session_repository.get_by_id(db, session_id)
        if row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session introuvable")
        notes = await course_section_note_repository.get_for_session(db, session_id)
        video_notes = await course_video_note_repository.get_for_session(db, session_id)

    gemini_response = row.gemini_response
    if gemini_response.get("sections"):
        gemini_response = {
            **gemini_response,
            "sections": [
                {
                    **s,
                    "note": notes.get(str(s.get("id")), ""),
                    "incomplete": is_incomplete_section_dict(s),
                }
                for s in gemini_response["sections"]
            ],
        }
    if gemini_response.get("videos"):
        gemini_response = {
            **gemini_response,
            "videos": [
                {**v, "note": video_notes.get(v.get("video_id"), "")} for v in gemini_response["videos"]
            ],
        }

    return CourseHistoryDetail(
        id=row.id,
        created_at=row.created_at.isoformat(),
        question=row.question,
        filenames=row.filenames,
        mode=row.mode,
        folder=row.folder,
        subfolder=row.subfolder,
        gemini_response=gemini_response,
    )


@router.delete("/courses/history/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_course_history(session_id: UUID, request: Request) -> None:
    """Supprime une session de cours et tout son contenu associé (podcast, révisions de
    flashcards, notes de section/vidéo — cascade via les FK). 404 si introuvable."""
    session_factory: async_sessionmaker = request.app.state.db_session_factory
    async with session_factory() as db:
        deleted = await course_session_repository.delete(db, session_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session introuvable")


@router.get("/courses/folders", response_model=list[FolderSummary])
async def list_course_folders(request: Request) -> list[FolderSummary]:
    """Dossiers/sous-dossiers utilisés par au moins un cours, avec leur nombre de cours.

    Un dossier n'est qu'un attribut de rangement porté par chaque cours (aucune entité dossier
    séparée, aucun dossier physique) : il n'apparaît ici qu'autant qu'au moins un cours y est
    rangé, et en sort dès que ce n'est plus le cas.
    """
    session_factory: async_sessionmaker = request.app.state.db_session_factory
    async with session_factory() as db:
        rows = await course_session_repository.list_folders(db)

    grouped: dict[str, dict[str, int]] = defaultdict(dict)
    for folder_name, subfolder_name, count in rows:
        grouped[folder_name][subfolder_name] = count

    return [
        FolderSummary(
            name=folder_name,
            course_count=sum(subfolder_counts.values()),
            subfolders=[
                SubfolderSummary(name=subfolder_name, course_count=count)
                for subfolder_name, count in subfolder_counts.items()
            ],
        )
        for folder_name, subfolder_counts in grouped.items()
    ]


@router.put("/courses/history/{session_id}/folder", response_model=CourseHistoryItem)
async def move_course_folder(
    session_id: UUID, body: MoveCourseFolderRequest, request: Request,
) -> CourseHistoryItem:
    """Déplace un cours vers un dossier/sous-dossier — créés implicitement s'ils n'existent pas
    encore (un dossier n'est qu'un attribut du cours). 404 si la session est introuvable."""
    session_factory: async_sessionmaker = request.app.state.db_session_factory
    async with session_factory() as db:
        row = await course_session_repository.move_to_folder(
            db, session_id, folder=body.folder, subfolder=body.subfolder or DEFAULT_COURSE_SUBFOLDER,
        )
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session introuvable")
    return CourseHistoryItem(
        id=row.id,
        created_at=row.created_at.isoformat(),
        question=row.question,
        filenames=row.filenames,
        mode=row.mode,
        folder=row.folder,
        subfolder=row.subfolder,
    )


@router.delete("/courses/folders/{folder_name}", response_model=FolderMoveResult)
async def delete_course_folder(folder_name: str, request: Request) -> FolderMoveResult:
    """Supprime un dossier : ses cours (et ceux de ses sous-dossiers) rejoignent le dossier par
    défaut (aucun cours n'est jamais supprimé par cette opération). 400 si `folder_name` est le
    dossier par défaut lui-même — il n'est jamais supprimable, c'est la racine de repli."""
    _reject_default_folder(folder_name)
    session_factory: async_sessionmaker = request.app.state.db_session_factory
    async with session_factory() as db:
        moved = await course_session_repository.delete_folder(db, folder_name)
    return FolderMoveResult(moved=moved)


@router.delete(
    "/courses/folders/{folder_name}/subfolders/{subfolder_name}", response_model=FolderMoveResult
)
async def delete_course_subfolder(
    folder_name: str, subfolder_name: str, request: Request,
) -> FolderMoveResult:
    """Supprime un sous-dossier : ses cours rejoignent le sous-dossier par défaut, dans le même
    dossier (aucun cours n'est jamais supprimé). 400 si `subfolder_name` est le sous-dossier par
    défaut lui-même."""
    _reject_default_subfolder(subfolder_name)
    session_factory: async_sessionmaker = request.app.state.db_session_factory
    async with session_factory() as db:
        moved = await course_session_repository.delete_subfolder(db, folder_name, subfolder_name)
    return FolderMoveResult(moved=moved)


def _reject_default_folder(folder_name: str) -> None:
    """400 si `folder_name` désigne le dossier par défaut (racine de repli, jamais supprimable)."""
    if folder_name.strip().casefold() == DEFAULT_COURSE_FOLDER.casefold():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Le dossier par défaut « {DEFAULT_COURSE_FOLDER} » ne peut pas être supprimé",
        )


def _reject_default_subfolder(subfolder_name: str) -> None:
    """400 si `subfolder_name` désigne le sous-dossier par défaut."""
    if subfolder_name.strip().casefold() == DEFAULT_COURSE_SUBFOLDER.casefold():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Le sous-dossier par défaut « {DEFAULT_COURSE_SUBFOLDER} » ne peut pas être supprimé",
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


def _limit_challenge(request: Request, settings: Settings = Depends(get_settings)) -> None:
    limiter = getattr(request.app.state, "challenge_rate_limiter", None)
    if limiter is None:
        limiter = request.app.state.challenge_rate_limiter = SlidingWindowLimiter()
    limiter.check(
        request.client.host if request.client else "unknown",
        settings.challenge_rate_limit_per_minute,
        "Trop d'analyses de défi, réessayez dans une minute",
    )


@router.post(
    "/courses/{session_id}/sections/{section_id}/challenge",
    response_model=ChallengeResponse,
    dependencies=[Depends(_limit_challenge)],
)
async def evaluate_section_challenge(
    session_id: UUID,
    section_id: str,
    body: ChallengeRequest,
    request: Request,
    gemini_client: GeminiClient = Depends(get_gemini_client),
) -> ChallengeResponse:
    """Analyse la réponse de l'apprenant au défi d'une section, avant l'explication.

    Le défi et ses idées attendues sont lus en base, jamais fournis par le client ; la réponse est
    une donnée (balisée, neutralisée) et n'est pas persistée. Le retour oriente vers l'explication
    sans la révéler. 404 si la session, la section ou son défi n'existe pas ; 422 si la réponse est
    vide ou dépasse 1000 caractères ; 429 au-delà de `challenge_rate_limit_per_minute` ; 502/503 si
    Gemini échoue.
    """
    session_factory: async_sessionmaker = request.app.state.db_session_factory
    async with session_factory() as db:
        row = await course_session_repository.get_by_id(db, session_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session introuvable")

    section = next(
        (s for s in (row.gemini_response.get("sections") or []) if str(s.get("id")) == section_id), None
    )
    if section is None or not (section.get("challenge") or "").strip():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Section introuvable")

    with _gemini_http_errors():
        evaluation = await evaluate_challenge(section, body.answer, gemini_client)
    return ChallengeResponse(verdict=evaluation.verdict.value, feedback=evaluation.feedback, hint=evaluation.hint)


def _limit_regenerate(request: Request, settings: Settings = Depends(get_settings)) -> None:
    limiter = getattr(request.app.state, "regenerate_rate_limiter", None)
    if limiter is None:
        limiter = request.app.state.regenerate_rate_limiter = SlidingWindowLimiter()
    limiter.check(
        request.client.host if request.client else "unknown",
        settings.course_regenerate_rate_limit_per_minute,
        "Trop de régénérations, réessayez dans une minute",
    )


@router.post(
    "/courses/{session_id}/sections/{section_id}/regenerate",
    response_model=CourseSection,
    dependencies=[Depends(_limit_regenerate)],
)
async def regenerate_course_section(
    session_id: UUID,
    section_id: str,
    request: Request,
    gemini_client: GeminiClient = Depends(get_gemini_client),
    settings: Settings = Depends(get_settings),
) -> CourseSection:
    """Régénère le contenu d'une section marquée `incomplete` (échec temporaire à la génération).

    404 si la session ou la section n'existe pas ; 409 si la section n'est pas incomplète (seules les
    sections en échec peuvent être régénérées) ; 429 au-delà de `course_regenerate_rate_limit_per_minute`.
    La note éventuelle de l'apprenant sur cette section est conservée (table séparée, non affectée).
    """
    session_factory: async_sessionmaker = request.app.state.db_session_factory
    async with session_factory() as db:
        row = await course_session_repository.get_by_id(db, session_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session introuvable")

    section = next(
        (s for s in (row.gemini_response.get("sections") or []) if str(s.get("id")) == section_id), None
    )
    if section is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Section introuvable")
    if not is_incomplete_section_dict(section):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Cette section n'est pas incomplète : seules les sections en échec peuvent être régénérées",
        )

    vector_store = request.app.state.vector_store
    with _gemini_http_errors():
        regenerated = await regenerate_section(row, section["title"], gemini_client, vector_store, settings)
    mapped = _map_sections_to_course_sections([regenerated])
    if not mapped:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Section régénérée vide")
    mapped = await resolve_visuals_in_sections(
        mapped, settings=settings, db_session_factory=session_factory, gemini_client=gemini_client,
    )
    updated = mapped[0].model_copy(update={"id": section_id})

    async with session_factory() as db:
        saved = await course_session_repository.update_section(db, session_id, section_id, updated.model_dump(mode="json"))
    if saved is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session ou section introuvable")
    return updated


def _limit_add_sections(request: Request, settings: Settings = Depends(get_settings)) -> None:
    limiter = getattr(request.app.state, "add_sections_rate_limiter", None)
    if limiter is None:
        limiter = request.app.state.add_sections_rate_limiter = SlidingWindowLimiter()
    limiter.check(
        request.client.host if request.client else "unknown",
        settings.add_course_sections_rate_limit_per_minute,
        "Trop de demandes d'ajout de contenu, réessayez dans une minute",
    )


@router.post(
    "/courses/{session_id}/sections",
    response_model=AddCourseSectionsResponse,
    dependencies=[Depends(_limit_add_sections)],
)
async def add_course_content(
    session_id: UUID,
    body: AddCourseSectionsRequest,
    request: Request,
    gemini_client: GeminiClient = Depends(get_gemini_client),
    settings: Settings = Depends(get_settings),
) -> AddCourseSectionsResponse:
    """Ajoute une ou plusieurs sections à un cours déjà généré.

    `instructions` vide : le contenu vient de `next_steps` si le cours en a, sinon de nouveaux
    sujets proposés par le modèle à partir du cours existant. La réponse contient `next_steps` mis
    à jour (pistes consommées retirées), que le client doit substituer à l'ancienne valeur.

    404 si la session n'existe pas ; 429 au-delà de `add_course_sections_rate_limit_per_minute`.
    """
    session_factory: async_sessionmaker = request.app.state.db_session_factory
    async with session_factory() as db:
        row = await course_session_repository.get_by_id(db, session_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session introuvable")

    vector_store = request.app.state.vector_store
    with _gemini_http_errors():
        new_sections, next_steps = await add_course_sections(
            row, body.instructions or "", gemini_client, vector_store, settings
        )
    start_index = len(row.gemini_response.get("sections") or [])
    mapped = _map_sections_to_course_sections(new_sections, start_index=start_index)
    if not mapped:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Aucune section générée")
    mapped = await resolve_visuals_in_sections(
        mapped, settings=settings, db_session_factory=session_factory, gemini_client=gemini_client,
    )

    async with session_factory() as db:
        saved = await course_session_repository.append_sections(
            db, session_id, [s.model_dump(mode="json") for s in mapped], next_steps
        )
    if saved is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session introuvable")
    return AddCourseSectionsResponse(sections=mapped, next_steps=next_steps)


def _limit_note(request: Request, settings: Settings = Depends(get_settings)) -> None:
    limiter = getattr(request.app.state, "note_rate_limiter", None)
    if limiter is None:
        limiter = request.app.state.note_rate_limiter = SlidingWindowLimiter()
    limiter.check(
        request.client.host if request.client else "unknown",
        settings.course_note_rate_limit_per_minute,
        "Trop de notes enregistrées, réessayez dans une minute",
    )


@router.put(
    "/courses/{session_id}/sections/{section_id}/note",
    response_model=SectionNoteResponse,
    dependencies=[Depends(_limit_note)],
)
async def save_section_note(
    session_id: UUID,
    section_id: str,
    body: SectionNoteRequest,
    request: Request,
) -> SectionNoteResponse:
    """Enregistre (ou efface, avec une note vide) la note libre de l'apprenant sur une section.

    404 si la session ou la section n'existe pas ; 429 au-delà de `course_note_rate_limit_per_minute`.
    """
    session_factory: async_sessionmaker = request.app.state.db_session_factory
    async with session_factory() as db:
        row = await course_session_repository.get_by_id(db, session_id)
        if row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session introuvable")
        if not any(str(s.get("id")) == section_id for s in (row.gemini_response.get("sections") or [])):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Section introuvable")

        saved = await course_section_note_repository.upsert(db, session_id, section_id, body.note)
    return SectionNoteResponse(note=saved.note, updated_at=saved.updated_at.isoformat())


@router.put(
    "/courses/{session_id}/videos/{video_id}/note",
    response_model=VideoNoteResponse,
    dependencies=[Depends(_limit_note)],
)
async def save_video_note(
    session_id: UUID,
    video_id: str,
    body: VideoNoteRequest,
    request: Request,
) -> VideoNoteResponse:
    """Enregistre (ou efface, avec une note vide) la note libre de l'apprenant sur une vidéo.

    404 si la session ou la vidéo n'existe pas ; 429 au-delà de `course_note_rate_limit_per_minute`
    (limite partagée avec les notes de section).
    """
    session_factory: async_sessionmaker = request.app.state.db_session_factory
    async with session_factory() as db:
        row = await course_session_repository.get_by_id(db, session_id)
        if row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session introuvable")
        if not any(v.get("video_id") == video_id for v in (row.gemini_response.get("videos") or [])):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Vidéo introuvable")

        saved = await course_video_note_repository.upsert(db, session_id, video_id, body.note)
    return VideoNoteResponse(note=saved.note, updated_at=saved.updated_at.isoformat())
