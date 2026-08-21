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


@router.post("/pdf/ingest", response_model=PDFIngestResponse)
async def ingest_pdf(
    request: Request,
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
) -> PDFIngestResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Un fichier PDF est requis")

    content = await file.read()
    chunks = extract_pdf_chunks(content, file.filename, settings)
    if not chunks:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Aucun contenu exploitable trouvé dans le PDF")

    vector_store = request.app.state.vector_store
    added = await vector_store.add_chunks(chunks)

    return PDFIngestResponse(
        status="ok",
        filename=file.filename,
        chunks_added=added,
        documents_added=len(chunks),
    )


@router.post("/pdf/search", response_model=DocumentQueryResponse)
async def search_pdf(
    request: Request,
    body: DocumentQueryRequest,
) -> DocumentQueryResponse:
    vector_store = request.app.state.vector_store
    results = await vector_store.search(body.query, top_k=body.top_k)
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
    except GeminiUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except GeminiQuotaExceededError as exc:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)) from exc
    except GeminiInvalidResponseError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc