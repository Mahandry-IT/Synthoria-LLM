import uuid
from datetime import date, datetime, timezone

from sqlalchemy import Boolean, Date, Float, ForeignKey, Index, Integer, Numeric, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base

# Dossier/sous-dossier ne sont qu'un attribut de rangement sur le cours (aucune entité dossier
# séparée, aucun dossier physique) : ces valeurs par défaut jouent le rôle de « racine » toujours
# existante, jamais supprimable — supprimer un dossier/sous-dossier n'efface aucun cours, ça y
# replace juste ses cours (voir course_session_repository.delete_folder/delete_subfolder).
# Les fichiers ingérés (`IngestedFile`) partagent ces mêmes valeurs par défaut.
DEFAULT_COURSE_FOLDER = "Général"
DEFAULT_COURSE_SUBFOLDER = "Non classé"


class CourseSession(Base):
    """Session de génération de cours, persistée en PostgreSQL."""

    __tablename__ = "course_sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=datetime.utcnow, nullable=False
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    filenames: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    mode: Mapped[str] = mapped_column(String(20), nullable=False)
    gemini_response: Mapped[dict] = mapped_column(JSONB, nullable=False)
    folder: Mapped[str] = mapped_column(String(200), nullable=False, default=DEFAULT_COURSE_FOLDER)
    subfolder: Mapped[str] = mapped_column(String(200), nullable=False, default=DEFAULT_COURSE_SUBFOLDER)

    __table_args__ = (
        Index("idx_course_sessions_created_at", created_at.desc()),
        Index("idx_course_sessions_folder", "folder", "subfolder"),
    )


class CoursePlan(Base):
    """Plan de cours proposé, en attente de validation par l'utilisateur.

    Le contexte de récupération (RAG / recherche web) est figé au moment du
    plan : le cours validé est généré à partir du même contexte, pas d'un
    nouveau retrieval potentiellement différent.
    """

    __tablename__ = "course_plans"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=datetime.utcnow, nullable=False
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    mode: Mapped[str] = mapped_column(String(20), nullable=False)
    # Mode de cours (express/standard/approfondi) ; migration 015, défaut serveur pour les anciens plans.
    depth: Mapped[str] = mapped_column(
        String(12), nullable=False, default="approfondi", server_default="approfondi"
    )
    filenames: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    top_k: Mapped[int] = mapped_column(Integer, nullable=False)
    full_document: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    retrieval_context: Mapped[dict] = mapped_column(JSONB, nullable=False)
    plan: Mapped[dict] = mapped_column(JSONB, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    expires_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)

    __table_args__ = (
        Index("idx_course_plans_expires_at", expires_at),
    )


class PodcastJob(Base):
    """Job asynchrone de génération de podcast à partir d'un cours persisté.

    Sert aussi de file d'attente : un worker réclame les jobs `pending` (ou
    périmés) via `SELECT ... FOR UPDATE SKIP LOCKED`. `script` est le
    checkpoint de l'étape de scriptage (reprise sans rappeler Gemini).
    """

    __tablename__ = "podcast_jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    course_session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("course_sessions.id", ondelete="CASCADE"), nullable=False
    )
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    stage: Mapped[str | None] = mapped_column(String(20), nullable=True)
    progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    params: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    params_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    script: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    audio_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    tts_engine: Mapped[str | None] = mapped_column(String(50), nullable=True)
    voices: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    locked_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    __table_args__ = (
        Index("idx_podcast_jobs_status_created", "status", "created_at"),
        Index("idx_podcast_jobs_session_params", "course_session_id", "params_hash"),
    )


class FlashcardReview(Base):
    """État de répétition espacée (Leitner) d'une flashcard d'un cours persisté."""

    __tablename__ = "flashcard_reviews"

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("course_sessions.id", ondelete="CASCADE"), primary_key=True
    )
    card_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    box: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    due_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    last_result: Mapped[str | None] = mapped_column(String(10), nullable=True)
    # Variante à présenter à la prochaine échéance (0 = dérivée des `check_questions`, non stockée).
    variant_no: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")

    __table_args__ = (Index("idx_flashcard_reviews_due_at", due_at),)


class YoutubeSearchCache(Base):
    """Résultats mis en cache d'une requête YouTube Data API (`search.list` + `videos.list` enrichis).

    `query_key` est un hash de la requête normalisée (+ langue, région) : indépendant du cours qui
    a déclenché la recherche, pour que deux cours proches partagent le cache.
    """

    __tablename__ = "youtube_search_cache"

    query_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    candidates: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    fetched_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)

    __table_args__ = (Index("idx_youtube_search_cache_fetched_at", fetched_at),)


class CourseSectionNote(Base):
    """Note libre de l'apprenant sur une section d'un cours persisté (pense-bête, idées).

    Distincte de `gemini_response` : jamais écrite par la génération, table séparée pour ne pas
    mélanger le contenu généré (snapshot du modèle) et l'annotation de l'apprenant.
    """

    __tablename__ = "course_section_notes"

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("course_sessions.id", ondelete="CASCADE"), primary_key=True
    )
    section_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    note: Mapped[str] = mapped_column(Text, nullable=False, default="")
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )


class CourseVideoNote(Base):
    """Note libre de l'apprenant sur une vidéo YouTube d'un cours persisté (pense-bête, idées).

    Même principe que `CourseSectionNote` : table séparée, jamais écrite par la génération. Les
    vidéos n'ont pas d'id propre en base (elles vivent dans `gemini_response.videos`, JSONB) ; on
    clé donc sur `video_id`, l'id YouTube, déjà dédupliqué par session à l'attache des vidéos.
    """

    __tablename__ = "course_video_notes"

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("course_sessions.id", ondelete="CASCADE"), primary_key=True
    )
    video_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    note: Mapped[str] = mapped_column(Text, nullable=False, default="")
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class CourseChatMessage(Base):
    """Message du chatbot d'un cours (question de l'apprenant ou réponse du tuteur).

    Table séparée de `gemini_response`, comme les notes : le chat n'altère jamais le contenu
    généré. Le quota journalier est dérivé de cette table (messages `user` depuis minuit UTC),
    sans table de compteur dédiée.
    """

    __tablename__ = "course_chat_messages"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("course_sessions.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)  # user | assistant
    content: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="answered")  # answered | off_topic
    sources: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), default=_utc_now, nullable=False)
    # Arbre des versions : parent d'une question = réponse précédente (NULL = racine), parent d'une
    # réponse = sa question. Des questions de même parent sont des versions l'une de l'autre.
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("course_chat_messages.id", ondelete="CASCADE"), nullable=True
    )
    # Suppression logique : le message reste compté dans le quota du jour.
    deleted_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)

    __table_args__ = (
        Index("idx_course_chat_messages_session_created", "session_id", "created_at"),
        Index("idx_course_chat_messages_parent", "parent_id"),
    )


class MediaAsset(Base):
    """Image ré-hébergée (jamais de hotlink) : téléchargée, ré-encodée, servie par `GET /media/{id}`.

    `sha256` est unique : deux blocs qui résolvent vers la même image (même figure PDF réutilisée,
    même image web) partagent une seule ligne (déduplication).
    """

    __tablename__ = "media_assets"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    kind: Mapped[str] = mapped_column(String(20), nullable=False)  # source | web | generated
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    mime: Mapped[str] = mapped_column(String(50), nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    storage_path: Mapped[str] = mapped_column(Text, nullable=False)
    alt_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    origin_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    author: Mapped[str | None] = mapped_column(Text, nullable=True)
    license: Mapped[str | None] = mapped_column(String(50), nullable=True)
    license_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_filename: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), default=datetime.utcnow, nullable=False)

    __table_args__ = (Index("idx_media_assets_created_at", created_at),)


class MediaQueryCache(Base):
    """Cache d'une recherche d'image web (Wikimedia Commons / Openverse) par requête normalisée.

    `asset_id` NULL = résultat négatif mis en cache (aucun candidat retenu) : un hit sur une ligne
    à `asset_id` NULL retire le bloc sans nouvel appel réseau ni Gemini, au même titre qu'un hit
    positif réutilise l'asset existant.
    """

    __tablename__ = "media_query_cache"

    query_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    asset_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("media_assets.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), default=datetime.utcnow, nullable=False)

    __table_args__ = (Index("idx_media_query_cache_created_at", created_at),)


class GeminiModelQuota(Base):
    """Consommation Gemini du jour par modèle, partagée entre les process `api` et `worker`
    (chacun a son propre `GeminiRateLimiter` en mémoire, non coordonné — cette table est la seule
    vue commune de ce qui a réellement été consommé, voir `app/services/gemini_quota_manager.py`).

    `day` est réinitialisée au prochain minuit Pacifique (`app/services/quota_reset.py`), alignée
    sur le fuseau horaire des quotas Google, indépendamment du fuseau système des conteneurs.
    """

    __tablename__ = "gemini_model_quota"

    model: Mapped[str] = mapped_column(String(100), primary_key=True)
    day: Mapped[date] = mapped_column(Date, nullable=False)
    request_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    exhausted_until: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), default=datetime.utcnow, nullable=False)

    __table_args__ = (Index("idx_gemini_model_quota_day", day),)


class GeminiResponseCache(Base):
    """Cache d'une réponse Gemini (`reformulate_query`/`describe_images`/`rank_images`) par hash de
    requête — réingérer un contenu déjà vu (ex. même PDF) évite un nouvel appel Gemini.

    `query_hash` dérive de la méthode + du contenu de la requête (prompt, hash des images, nom du
    schema) — voir `app/repositories/gemini_response_cache_repository.py`.
    """

    __tablename__ = "gemini_response_cache"

    query_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    method: Mapped[str] = mapped_column(String(50), nullable=False)
    response: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), default=datetime.utcnow, nullable=False)

    __table_args__ = (Index("idx_gemini_response_cache_created_at", created_at),)


class IngestedFile(Base):
    """Rangement (dossier/sous-dossier) d'un fichier ingéré dans le vector store.

    Le vector store JSON ne connaît que `filename` : le classement vit ici pour qu'un déplacement
    ne réécrive jamais le store. Un fichier sans ligne est rangé dans le dossier par défaut (mêmes
    valeurs que les cours) — aucune ligne n'est donc créée tant qu'on ne le déplace pas.
    """

    __tablename__ = "ingested_files"

    filename: Mapped[str] = mapped_column(String(512), primary_key=True)
    folder: Mapped[str] = mapped_column(String(200), nullable=False, default=DEFAULT_COURSE_FOLDER)
    subfolder: Mapped[str] = mapped_column(String(200), nullable=False, default=DEFAULT_COURSE_SUBFOLDER)
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    __table_args__ = (Index("idx_ingested_files_folder", "folder", "subfolder"),)


class QuizBankQuestion(Base):
    """Question de la banque de QCM d'un cours (lot 0 = quiz d'origine, lots suivants = recharges).

    `payload` est la question complète au format d'API (`QuizQuestion`), bonnes réponses comprises :
    elle n'est jamais renvoyée telle quelle au démarrage d'une tentative.
    """

    __tablename__ = "quiz_questions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("course_sessions.id", ondelete="CASCADE"), nullable=False
    )
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    batch: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    times_served: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_served_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), default=_utc_now, nullable=False)

    __table_args__ = (Index("idx_quiz_questions_session_served", "session_id", "times_served"),)


class QuizAttempt(Base):
    """Tentative de quiz notée côté serveur (`in_progress` → `completed` | `aborted`)."""

    __tablename__ = "quiz_attempts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("course_sessions.id", ondelete="CASCADE"), nullable=False
    )
    question_ids: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    answers: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    score: Mapped[float | None] = mapped_column(Numeric(5, 2, asdecimal=False), nullable=True)
    max_score: Mapped[float] = mapped_column(Numeric(5, 2, asdecimal=False), nullable=False, default=20)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="in_progress")
    abort_reason: Mapped[str | None] = mapped_column(String(200), nullable=True)
    started_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), default=_utc_now, nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)

    __table_args__ = (Index("idx_quiz_attempts_session_started", "session_id", "started_at"),)


class FlashcardVariant(Base):
    """Variante générée d'une carte de révision (même notion, autre formulation), numérotée à
    partir de 1 ; la variante 0 reste celle dérivée des `check_questions` du cours."""

    __tablename__ = "flashcard_variants"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("course_sessions.id", ondelete="CASCADE"), nullable=False
    )
    card_id: Mapped[str] = mapped_column(String(64), nullable=False)
    variant_no: Mapped[int] = mapped_column(Integer, nullable=False)
    front: Mapped[str] = mapped_column(Text, nullable=False)
    choices: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    correct_indices: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    explanation: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), default=_utc_now, nullable=False)

    __table_args__ = (
        UniqueConstraint("session_id", "card_id", "variant_no", name="uq_flashcard_variants_card_variant"),
    )
