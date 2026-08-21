from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status

from app.api.schemas import (
    COURSE_DEFAULT_QUESTION,
    CourseGenerationRequest,
    CourseGenerationResponse,
    DocumentQueryRequest,
    DocumentQueryResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    PDFIngestMultiResponse,
    PDFIngestResponse,
)
from app.core.config import Settings, get_settings
from app.core.exceptions import (
    GeminiInvalidResponseError,
    GeminiQuotaExceededError,
    GeminiServiceError,
    GeminiUnavailableError,
    OllamaModelNotFoundError,
    OllamaUnavailableError,
)
from app.services.course_generator import generate_course_from_question
from app.services.gemini_client import GeminiClient
from app.services.ollama_client import OllamaClient
from app.services.pdf_pipeline import extract_pdf_chunks

router = APIRouter()


def get_ollama_client(request: Request) -> OllamaClient:
    return request.app.state.ollama_client


def get_gemini_client(request: Request) -> GeminiClient:
    return request.app.state.gemini_client


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

        added = await request.app.state.vector_store.add_chunks(chunks)
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
    # Si un filtre filename est fourni, on sur-recherche pour compenser
    # le post-filtrage qui peut éliminer des résultats
    search_k = body.top_k * 3 if body.filename else body.top_k
    raw_results = await vector_store.search(body.query, top_k=search_k)

    if body.filename:
        allowed = {body.filename} if isinstance(body.filename, str) else set(body.filename)
        results = [
            r for r in raw_results
            if r.get("metadata", {}).get("filename") in allowed
        ][:body.top_k]
    else:
        results = raw_results[:body.top_k]

    return DocumentQueryResponse(query=body.query, results=results)


@router.post("/courses/generate", response_model=CourseGenerationResponse)
async def generate_course(
    request: Request,
    body: CourseGenerationRequest,
    gemini_client: GeminiClient = Depends(get_gemini_client),
    settings: Settings = Depends(get_settings),
) -> CourseGenerationResponse:
    """Mode 2 (fichier + question) ou Mode 3 (question seule + recherche web)."""
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
        return await generate_course_from_question(
            question=question,
            vector_store=vector_store,
            gemini_client=gemini_client,
            settings=settings,
            mode=resolved_mode,
            top_k=body.top_k,
            filename=body.filename,
        )
    except GeminiQuotaExceededError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "quota_exceeded",
                "gemini_code": exc.error_code,
                "gemini_status": "RESOURCE_EXHAUSTED",
                "message": exc.error_message or str(exc),
            },
        ) from exc
    except GeminiUnavailableError as exc:
        # Mapper les codes Gemini vers les HTTP status appropriés
        gemini_code = exc.error_code
        if gemini_code == 400:
            http_status = status.HTTP_400_BAD_REQUEST
        elif gemini_code == 401 or gemini_code == 403:
            http_status = status.HTTP_502_BAD_GATEWAY
        elif gemini_code == 404:
            http_status = status.HTTP_502_BAD_GATEWAY
        else:
            http_status = status.HTTP_503_SERVICE_UNAVAILABLE
        raise HTTPException(
            status_code=http_status,
            detail={
                "error": "gemini_unavailable",
                "gemini_code": gemini_code,
                "message": exc.error_message or str(exc),
            },
        ) from exc
    except GeminiInvalidResponseError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "error": "invalid_response",
                "message": str(exc),
            },
        ) from exc