from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration de l'application, chargée depuis les variables d'environnement."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = "Synthoria LLM"
    ollama_base_url: str = "http://ollama:11434"
    ollama_default_model: str = "llama3.2"
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_timeout_seconds: float = 60.0
    ollama_max_retries: int = 3
    request_max_prompt_length: int = 8000
    cors_allowed_origins: list[str] = ["*"]
    rate_limit_per_minute: int = 30
    chroma_persist_directory: str = "./chroma_db"
    chroma_collection_name: str = "synthoria_documents"
    pdf_chunk_target_tokens: int = 400
    pdf_chunk_overlap_tokens: int = 50
    pdf_chunk_min_tokens: int = 80
    course_full_document_mode: bool = False
    gemini_api_key: str | None = None
    gemini_model_flash: str = "gemini-3.6-flash"
    gemini_model_flash_lite: str = "gemini-3.5-flash-lite"
    gemini_max_retries: int = 3
    gemini_timeout_seconds: float = 30.0
    course_top_k_default: int = 6
    course_question_max_length: int = 2000
    gemini_use_search_grounding: bool = True
    database_url: str = "postgresql+asyncpg://synthoria:synthoria@postgres:5432/synthoria"


@lru_cache
def get_settings() -> Settings:
    return Settings()