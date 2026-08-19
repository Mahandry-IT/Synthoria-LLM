import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import get_settings
from app.core.rate_limit import RateLimitMiddleware
from app.services.gemini_client import GeminiClient
from app.services.ollama_client import OllamaClient
from app.services.vector_store import NumpyVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.ollama_client = OllamaClient(settings)
    app.state.vector_store = NumpyVectorStore(settings, app.state.ollama_client)
    app.state.gemini_client = GeminiClient(settings)
    yield
    await app.state.ollama_client.close()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allowed_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.add_middleware(RateLimitMiddleware, requests_per_minute=settings.rate_limit_per_minute)

    app.include_router(router)
    return app


app = create_app()