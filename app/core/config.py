from datetime import timedelta
from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration de l'application, chargée depuis les variables d'environnement."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Synthoria LLM"
    ollama_base_url: str = "http://ollama:11435"
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
    # Marge sous la limite Google (ex. 15 RPM sur gemini-*-flash-lite au palier gratuit) : les
    # appels sont espacés pour rester sous ce seuil plutôt que de heurter un 429. Par processus
    # (api et worker ont chacun leur fenêtre) — voir app/services/gemini_rate_limit.py.
    gemini_rpm_limit: int = 14
    course_top_k_default: int = 6
    course_question_max_length: int = 2000
    course_coverage_completion_enabled: bool = True
    course_coverage_min_missing_chars: int = 300
    # Sections plus lourdes (défi, exemple à trous, questions, reformulation) : lots plus petits.
    course_plan_batch_size: int = 2
    course_plan_ttl_minutes: int = 120
    course_videos_enabled: bool = True
    # Cible, jamais garantie : YouTube peut renvoyer moins de résultats pertinents que ce plancher.
    course_videos_min: int = 5
    course_videos_max: int = 10
    course_videos_verify_timeout_seconds: float = 5.0
    gemini_use_search_grounding: bool = True
    database_url: str = "postgresql+asyncpg://synthoria:synthoria@postgres:5432/synthoria"

    # Podcast (génération audio à partir d'un cours)
    podcast_enabled: bool = True
    podcast_auto_generate: bool = False
    podcast_storage_dir: str = "/data/podcasts"
    podcast_tts_base_url: str = "http://piper:5000"
    podcast_tts_timeout_seconds: float = 60.0
    podcast_tts_max_retries: int = 3
    podcast_tts_max_chars: int = 400
    podcast_voice_host: str = "fr_FR-upmc-medium:0"      # « modèle[:locuteur] »
    podcast_voice_expert: str = "fr_FR-upmc-medium:1"
    podcast_default_target_minutes: int = 10
    podcast_max_minutes: int = 60
    podcast_max_segments: int = 40
    podcast_script_batch_size: int = 4
    podcast_tts_concurrency: int = 2
    podcast_worker_poll_seconds: float = 5.0
    podcast_job_stale_minutes: int = 20
    podcast_max_attempts: int = 3
    podcast_audio_bitrate: str = "96k"
    podcast_retention_days: int = 30
    podcast_generate_rate_limit_per_minute: int = 5

    # Part maximale (0-1) de la réponse directe recopiée de l'introduction avant relance ciblée.
    course_answer_intro_similarity_max: float = 0.5

    # Part minimale de questions normale/difficile dans le quiz final (les faciles vont dans les sections).
    course_quiz_min_hard_share: float = 0.6

    # Évaluation de la reformulation (« explique avec tes mots »)
    recall_answer_max_length: int = 1000
    recall_rate_limit_per_minute: int = 10
    more_sections_rate_limit_per_minute: int = 6

    # Répétition espacée (Leitner) : intervalles en jours par boîte, cours analysés pour « à réviser »
    review_intervals_days: list[int] = [1, 3, 7, 21]
    review_sessions_scan_limit: int = 50
    review_rate_limit_per_minute: int = 60

    # Régénération d'une section incomplète, et notes libres de l'apprenant sur chaque section
    course_regenerate_rate_limit_per_minute: int = 6
    course_note_max_length: int = 2000
    course_note_rate_limit_per_minute: int = 20

    # YouTube Data API v3 : recherche réelle de vidéos (remplace les IDs inventés par Gemini).
    # Sans clé, comportement inchangé (repli sur le grounding Gemini + vérification oEmbed).
    youtube_api_key: SecretStr | None = None
    youtube_timeout_seconds: float = 5.0
    youtube_search_max_results: int = 10
    youtube_relevance_language: str = "fr"
    youtube_region_code: str = "FR"
    youtube_min_duration_seconds: int = 240
    youtube_max_duration_seconds: int = 2400
    youtube_max_queries: int = 2
    youtube_cache_ttl_hours: int = 72

    # V2 : classement / catégorisation pédagogique des candidats (1 appel Flash-Lite de plus par cours).
    course_videos_ranking_enabled: bool = True
    youtube_ranking_min_score: int = 40

    @property
    def youtube_cache_ttl(self) -> timedelta:
        return timedelta(hours=self.youtube_cache_ttl_hours)

    # Supports visuels : images ré-hébergées (jamais de hotlink), servies par GET /media/{id}.
    media_storage_dir: str = "/data/media"
    media_max_bytes: int = 5 * 1024 * 1024
    media_resolve_concurrency: int = 4
    media_resolve_timeout_seconds: float = 20.0


@lru_cache
def get_settings() -> Settings:
    return Settings()