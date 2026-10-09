import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat_routes import router as chat_router
from app.api.media_routes import router as media_router
from app.api.podcast_routes import router as podcast_router
from app.api.review_routes import router as review_router
from app.api.routes import router
from app.core.config import get_settings
from app.core.errors import register_error_handlers
from app.core.rate_limit import RateLimitMiddleware
from app.core.request_id import RequestIdLogFilter, RequestIdMiddleware
from app.db.base import Base
from app.db.session import create_engine
from app.db.models import CoursePlan, CourseSession, PodcastJob  # noqa: F401 — ensure Base.metadata knows the models
from app.db.schema_sync import sync_added_columns
from app.services.gemini_client import GeminiClient
from app.services.ollama_client import OllamaClient
from app.services.vector_store import NumpyVectorStore

_log_handler = logging.StreamHandler(sys.stdout)
_log_handler.addFilter(RequestIdLogFilter())
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s [%(request_id)s] %(message)s",
    handlers=[_log_handler],
    force=True,
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.ollama_client = OllamaClient(settings)
    app.state.vector_store = NumpyVectorStore(settings, app.state.ollama_client)

    # PostgreSQL async engine + auto-create tables — créé avant GeminiClient pour lui injecter
    # session_factory (état de quota partagé api/worker, voir gemini_quota_manager.py).
    engine, session_factory = create_engine(settings)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await sync_added_columns(conn)
    app.state.db_engine = engine
    app.state.db_session_factory = session_factory

    app.state.gemini_client = GeminiClient(settings, session_factory=session_factory)

    yield

    await engine.dispose()
    await app.state.ollama_client.close()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allowed_origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "Retry-After"],
    )
    app.add_middleware(RateLimitMiddleware, requests_per_minute=settings.rate_limit_per_minute)
    # Ajouté en dernier = le plus externe : l'identifiant existe déjà pour le limiteur et le CORS.
    app.add_middleware(RequestIdMiddleware)

    register_error_handlers(app, include_debug=settings.is_development)

    app.include_router(router)
    app.include_router(podcast_router)
    app.include_router(review_router)
    app.include_router(media_router)
    app.include_router(chat_router)
    return app


app = create_app()