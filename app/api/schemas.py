from datetime import datetime
from typing import Annotated, Any, Generic, Literal, TypeVar
from uuid import UUID

from fastapi import Query
from pydantic import BaseModel, Field, field_validator, model_validator

T = TypeVar("T")


class PageParams:
    """FastAPI dependency — paramètres de pagination standardisés."""

    def __init__(
        self,
        page: int = Query(1, ge=1, description="Numéro de page (commence à 1)"),
        limit: int = Query(20, ge=1, le=100, description="Nombre d'éléments par page (1-100)"),
    ) -> None:
        self.page = max(page, 1)
        self.limit = min(limit, 100)

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.limit


class PaginationMeta(BaseModel):
    page: int = Field(..., description="Page courante")
    limit: int = Field(..., description="Taille de page")
    total: int = Field(..., description="Nombre total d'éléments")
    totalPages: int = Field(..., description="Nombre total de pages")


class PaginatedResponse(BaseModel, Generic[T]):
    """Réponse paginée générique."""
    status: str = "ok"
    data: list[T]
    meta: PaginationMeta


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Prompt utilisateur")
    model: str | None = Field(None, description="Modèle Ollama à utiliser (sinon défaut config)")
    stream: bool = Field(False, description="Activer le streaming de la réponse")

    @field_validator("prompt")
    @classmethod
    def strip_prompt(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("prompt ne peut pas être vide")
        return v


class GenerateResponse(BaseModel):
    model: str
    response: str
    done: bool


class HealthResponse(BaseModel):
    status: str
    ollama_reachable: bool


class DocumentQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Question ou requête sur le PDF")
    top_k: int = Field(5, ge=1, le=20, description="Nombre de résultats vectoriels à renvoyer")
    filename: str | list[str] | None = Field(
        None,
        description="Filtre optionnel sur un ou plusieurs noms de fichier ingérés.",
    )


class PDFIngestResponse(BaseModel):
    status: str
    filename: str
    chunks_added: int
    documents_added: int
    message: str | None = Field(None, description="Message d'information (ex: doublon détecté).")


class PDFIngestMultiResponse(BaseModel):
    """Réponse agrégée pour l'ingestion multi-fichiers."""
    status: str
    files: list[PDFIngestResponse]
    total_chunks: int
    total_documents: int


class FileInfo(BaseModel):
    """Informations sur un fichier stocké dans le vector store."""
    id: int = Field(..., description="Identifiant séquentiel du fichier (1, 2, 3...)")
    filename: str = Field(..., description="Nom du fichier PDF")


class FileListResponse(BaseModel):
    """Réponse de l'endpoint GET /pdf/files (paginée)."""
    status: str = "ok"
    data: list[FileInfo]
    meta: PaginationMeta



class DocumentQueryResponse(BaseModel):
    query: str
    results: list[dict]


# --- Génération de cours structuré (Mode 2 : fichier + question) ---------------
# Enveloppe commune aux 3 modes (file_only / file_question / question_only) :
# `mode` et `sources` sont toujours présents pour la traçabilité et le routage
# front-end. `format` détermine la richesse du contenu attendu.


class Step(BaseModel):
    id: str = Field(..., description="Identifiant de l'étape, numérique sous forme de chaîne, ex. '1', '2' (clé React côté front)")
    content: str = Field(..., description="Contenu de l'étape")


class WorkedExample(BaseModel):
    statement: str = Field(..., description="Énoncé de l'exemple travaillé")
    steps: list[Step] = Field(..., description="Étapes de résolution détaillées")
    result: str = Field(..., description="Résultat final commenté")


class CourseSource(BaseModel):
    type: Literal["file", "web"]
    label: str
    reference: str


class CourseSection(BaseModel):
    id: str
    title: str
    quoi: str
    pourquoi: str
    comment: str
    worked_example: WorkedExample
    key_points: list[str] = Field(default_factory=list)


class CoursePitfall(BaseModel):
    description: str
    why_it_happens: str
    how_to_avoid: str


class QuizQuestion(BaseModel):
    """Question de quiz — rétro-compatible avec les anciennes données stockées.

    Les anciennes données en base peuvent avoir :
    - correct_option_index (int) au lieu de correct_option_indices (list[int])
    - pas de champ difficulty, points, time_limit_seconds
    Le model_validator ci-dessous harmonise ces cas.
    """

    question: str
    options: list[str] = Field(..., min_length=2)
    correct_option_indices: list[int] = Field(default=[], min_length=0, description="Indices 0-based des bonnes réponses (1 = unique, >1 = QCM multiple)")
    difficulty: Literal["facile", "normale", "difficile"] = "normale"
    points: float = Field(default=1.0, ge=0.0, description="Points alloués à cette question (calculé côté serveur, borne sup ~2.0 pour N≥10)")
    explanation: str = ""
    time_limit_seconds: int = Field(default=45, description="45 par défaut, 80 si la question implique un calcul")

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_fields(cls, values: dict) -> dict:
        """Rétro-compatibilité : convertit correct_option_index (int) → correct_option_indices (list)."""
        if isinstance(values, dict):
            # Ancien champ singular → nouveau champ pluriel
            if "correct_option_index" in values and "correct_option_indices" not in values:
                idx = values.pop("correct_option_index")
                values["correct_option_indices"] = [idx] if isinstance(idx, int) else idx
        return values

    @field_validator("correct_option_indices")
    @classmethod
    def _check_indices(cls, v: list[int], info) -> list[int]:
        if len(v) != len(set(v)):
            raise ValueError("correct_option_indices contient des doublons")
        options = info.data.get("options")
        if options:
            for idx in v:
                if idx < 0 or idx >= len(options):
                    raise ValueError(
                        f"correct_option_index {idx} hors bornes "
                        f"(options a {len(options)} éléments, index 0..{len(options)-1})"
                    )
        return v


class CourseMeta(BaseModel):
    title: str
    subject: str
    language: str = "fr"
    generated_at: str


class CourseAnswer(BaseModel):
    quoi: str
    pourquoi: str
    comment: str
    worked_example: WorkedExample
    key_points: list[str] = Field(default_factory=list)


COURSE_DEFAULT_QUESTION = "Explique moi le cours en complet"


class CourseGenerationRequest(BaseModel):
    question: str | None = Field(None, description="Question de l'utilisateur")
    mode: Literal["file_question", "question_only"] | None = Field(
        None,
        description=(
            "Mode de génération. Si non fourni, auto-détecté : "
            "file_question si filename est présent, question_only sinon."
        ),
    )
    top_k: int = Field(20, ge=1, le=20, description="Nombre de chunks à récupérer pour le contexte")
    filename: str | list[str] | None = Field(
        None,
        description=(
            "Filtre optionnel sur un ou plusieurs documents déjà ingérés. "
            "Accepte un seul nom de fichier (string) ou une liste de noms."
        ),
    )
    full_document: bool = Field(
        False,
        description=(
            "Si True, ignore top_k et récupère l'intégralité des chunks du/des "
            "fichier(s) filtré(s) pour une couverture exhaustive du document "
            "(augmente la taille du contexte envoyé à Gemini)."
        ),
    )

    generate_podcast: bool | None = Field(
        None,
        description="Lance la génération d'un podcast après le cours. Défaut : PODCAST_AUTO_GENERATE.",
    )


class CourseGenerationResponse(BaseModel):
    mode: Literal["file_only", "file_question", "question_only"]
    format: Literal["full_course", "focused_answer"]
    meta: CourseMeta
    sources: list[CourseSource]

    # Uniquement pour format == "full_course"
    introduction: dict[str, str] | None = None

    # Uniquement pour format == "focused_answer"
    answer: CourseAnswer | None = None

    # Uniquement pour format == "full_course"
    sections: list[CourseSection] | None = None
    common_pitfalls: list[CoursePitfall] | None = None
    quiz: list[QuizQuestion] | None = None

    summary: str
    next_steps: list[str] = Field(default_factory=list)

    # Renseignés par les routes après persistance (jamais stockés dans `gemini_response`).
    session_id: UUID | None = Field(None, description="Id de la session persistée (None si la persistance a échoué).")
    podcast_job_id: UUID | None = Field(None, description="Id du job podcast créé automatiquement, le cas échéant.")

    @model_validator(mode="after")
    def check_format_consistency(self) -> "CourseGenerationResponse":
        if self.format == "full_course" and (self.introduction is None or self.sections is None):
            raise ValueError("format='full_course' nécessite 'introduction' et 'sections'")
        if self.format == "focused_answer" and self.answer is None:
            raise ValueError("format='focused_answer' nécessite 'answer'")
        return self


# --- Génération en deux temps : plan validable → cours complet ----------------

# Plafond anti-abus (coût/quota Gemini) sur le plan renvoyé par le client.
# Ce n'est PAS une limite pédagogique : la génération du plan elle-même n'a pas de cap.
COURSE_PLAN_MAX_SECTIONS = 80
_PLAN_TITLE_MAX = 200
_PLAN_OBJECTIVE_MAX = 1000
_PLAN_SUBTOPIC_MAX = 300
_PLAN_SUBTOPICS_MAX_ITEMS = 20

PlannedSectionType = Literal["introduction", "development", "common_pitfalls", "summary", "next_steps"]


class CoursePlanRequest(CourseGenerationRequest):
    """Requête de planification : mêmes paramètres que la génération directe."""


class ApiPlannedSection(BaseModel):
    type: PlannedSectionType
    title: str = Field(..., min_length=1, max_length=_PLAN_TITLE_MAX)
    objective: str = Field("", max_length=_PLAN_OBJECTIVE_MAX)
    subtopics: list[Annotated[str, Field(min_length=1, max_length=_PLAN_SUBTOPIC_MAX)]] = Field(
        default_factory=list, max_length=_PLAN_SUBTOPICS_MAX_ITEMS
    )
    order: int = Field(..., ge=1)

    @field_validator("title", "objective")
    @classmethod
    def _strip_text(cls, v: str) -> str:
        return v.strip()

    @field_validator("title")
    @classmethod
    def _title_not_blank(cls, v: str) -> str:
        if not v:
            raise ValueError("title ne peut pas être vide")
        return v


class CoursePlanMeta(BaseModel):
    title: str
    subject: str
    language: str = "fr"


class CoursePlanResponse(BaseModel):
    plan_id: UUID
    expires_at: str
    mode: Literal["file_question", "question_only"]
    meta: CoursePlanMeta
    sections: list[ApiPlannedSection]
    coverage_notes: str = ""


_PLAN_INSTRUCTIONS_MAX = 1000


class RefineSectionRequest(BaseModel):
    """Demande de complétion d'une section de plan jugée incomplète par l'utilisateur.

    `sections` est le plan complet courant (contexte anti-doublon, facultatif) ;
    `instructions` est ce que l'utilisateur souhaite voir ajouté (facultatif :
    sans précision, le modèle détermine lui-même ce qui manque).
    """

    plan_id: UUID
    section: ApiPlannedSection
    sections: list[ApiPlannedSection] = Field(default_factory=list, max_length=COURSE_PLAN_MAX_SECTIONS)
    instructions: str | None = Field(None, max_length=_PLAN_INSTRUCTIONS_MAX)

    @field_validator("instructions")
    @classmethod
    def _blank_instructions_to_none(cls, v: str | None) -> str | None:
        v = (v or "").strip()
        return v or None


class MoreSectionsRequest(BaseModel):
    """Demande de nouvelles sections de développement à partir de « Pour aller plus loin »."""

    plan_id: UUID
    sections: list[ApiPlannedSection] = Field(..., min_length=1, max_length=COURSE_PLAN_MAX_SECTIONS)


class MoreSectionsResponse(BaseModel):
    sections: list[ApiPlannedSection]


class PendingPlanItem(BaseModel):
    """Plan non expiré, pas encore transformé en cours (liste du dashboard)."""

    plan_id: UUID
    question: str
    title: str
    subject: str
    sections_count: int
    created_at: str
    expires_at: str


class CoursePlanDetail(CoursePlanResponse):
    """Plan persisté relu tel que proposé (reprise depuis le dashboard), avec la requête d'origine."""

    question: str
    filenames: list[str] = Field(default_factory=list)


class CourseFromPlanRequest(BaseModel):
    """Plan (éventuellement édité par l'utilisateur) à transformer en cours complet.

    La question, le mode et les fichiers sont lus depuis le plan persisté
    (`plan_id`) — jamais depuis le client — pour garantir que le cours est
    généré à partir du contexte exact qui a servi à proposer le plan.
    """

    plan_id: UUID
    sections: list[ApiPlannedSection] = Field(..., min_length=1, max_length=COURSE_PLAN_MAX_SECTIONS)
    generate_podcast: bool | None = Field(
        None,
        description="Lance la génération d'un podcast après le cours. Défaut : PODCAST_AUTO_GENERATE.",
    )

    @model_validator(mode="after")
    def _check_has_development_section(self) -> "CourseFromPlanRequest":
        if not any(s.type == "development" for s in self.sections):
            raise ValueError("Le plan doit contenir au moins une section 'development'")
        return self


# --- Historique des sessions de cours (PostgreSQL) ---


class CourseHistoryItem(BaseModel):
    """Élément de la liste paginée de l'historique (sans gemini_response)."""
    id: UUID
    created_at: str
    question: str
    filenames: list[str]
    mode: str


class CourseHistoryDetail(BaseModel):
    """Détail d'une session de cours (avec gemini_response)."""
    id: UUID
    created_at: str
    question: str
    filenames: list[str]
    mode: str
    gemini_response: CourseGenerationResponse

# ─── Podcast ────────────────────────────────────────────────

PodcastStyle = Literal["conversational", "educational", "concise"]
PodcastJobState = Literal["pending", "scripting", "synthesizing", "mixing", "done", "failed"]


class PodcastGenerationRequest(BaseModel):
    style: PodcastStyle = Field("conversational", description="Ton du podcast.")
    target_minutes: int | None = Field(
        None, ge=3, le=60, description="Durée cible en minutes (défaut : configuration serveur)."
    )
    force: bool = Field(False, description="Crée un nouveau job même s'il en existe un équivalent non échoué.")


class PodcastJobResponse(BaseModel):
    job_id: UUID
    status: PodcastJobState


class PodcastJobStatus(BaseModel):
    job_id: UUID
    course_session_id: UUID
    status: PodcastJobState
    stage: str | None = None
    progress: int = Field(0, ge=0, le=100)
    error_message: str | None = None
    duration_seconds: float | None = None
    created_at: datetime
    updated_at: datetime


class PodcastJobList(BaseModel):
    data: list[PodcastJobStatus]


class PodcastSummary(PodcastJobStatus):
    """Podcast d'un cours pour les listes (dashboard) : statut du job + titre affichable."""

    title: str


class PodcastSummaryList(BaseModel):
    data: list[PodcastSummary]
