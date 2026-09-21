"""Pydantic schema for Gemini `response_schema` — structured pass of the
What/Why/How teacher (see course_generation_instructions.md).

v2: content is now block-based (sections > subsections > typed blocks)
instead of flat quoi/pourquoi/comment strings, so both `full_course` and
`focused_answer` formats share the same rendering model on the frontend.

Only Mode 2 (file_question) and Mode 3 (question_only) call the LLM.
Mode 1 (file-only) is pure ingestion → no schema needed.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, model_validator


class InteractionMode(str, Enum):
    FILE_QUESTION = "file_question"   # RAG context, no google_search
    QUESTION_ONLY = "question_only"   # google_search, two-call pattern


class OutputFormat(str, Enum):
    FULL_COURSE = "full_course"       # reserved for a future file-only generation trigger
    FOCUSED_ANSWER = "focused_answer"  # Mode 2 / Mode 3


class SectionType(str, Enum):
    INTRODUCTION = "introduction"
    DEVELOPMENT = "development"
    COMMON_PITFALLS = "common_pitfalls"
    SUMMARY = "summary"
    NEXT_STEPS = "next_steps"


class BlockType(str, Enum):
    TEXT = "text"
    DEFINITION = "definition"
    LIST = "list"
    TABLE = "table"
    FORMULA = "formula"
    CODE = "code"
    WORKED_EXAMPLE = "worked_example"
    CALLOUT = "callout"
    IMAGE = "image"
    PITFALL = "pitfall"


class CalloutVariant(str, Enum):
    NOTE = "note"
    WARNING = "warning"
    TIP = "tip"


class SourceType(str, Enum):
    FILE_CHUNK = "file_chunk"
    WEB = "web"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Source(BaseModel):
    type: SourceType = Field(description="Origin of this source: a retrieved file chunk or a web search result.")
    label: str = Field(description="Human-readable label (heading, page title, article title).")
    reference: str = Field(description="Chunk id (e.g. 'doc_3_chunk_12'), page number, or URL.")


class TableData(BaseModel):
    caption: str = Field(default="", description="Short title of the table (what it compares or summarizes).")
    headers: list[str] = Field(description="Column headers, in order.")
    rows: list[list[str]] = Field(description="Row values, each row matching the headers length.")


class VideoSuggestion(BaseModel):
    url: str = Field(description="Full YouTube URL (https://www.youtube.com/watch?v=...) of a video that really exists.")
    title: str = Field(default="", description="Title of the video.")


class FormulaData(BaseModel):
    latex: str = Field(description="Formula in LaTeX, exact — never round or simplify silently.")
    description: str | None = Field(default=None, description="Short plain-language reading of the formula.")


class WorkedExample(BaseModel):
    statement: str = Field(description="Concrete, non-placeholder statement of the example (with real numbers/data).")
    steps: list[str] = Field(
        description=(
            "Explicit intermediate steps, in order. Never skip to the final result. "
            "Programming code (only when the subject is programming) must be wrapped in single backticks (`code`); "
            "math and numbers are never in backticks — use $...$ for math, plain text for numbers, "
            "one whole statement per step — never split a statement across steps. "
            "Sentences end with a period, but never add a period or a closing "
            "parenthesis after code, and never wrap code in '(ex: ...)'."
        )
    )
    result: str = Field(description="Final result, commented — what it means, not just the raw value.")


class PitfallData(BaseModel):
    description: str = Field(description="The mistake itself — what learners commonly get wrong.")
    why_it_happens: str = Field(description="Root cause: why learners fall into this, not just that they do.")
    how_to_avoid: str = Field(description="Concrete corrective action or check to prevent it.")


class ContentBlock(BaseModel):
    """Single typed content unit. Only the field(s) matching `type` are populated."""

    type: BlockType = Field(description="Kind of content this block carries; determines which field below is set.")

    text: str | None = Field(
        default=None,
        description=(
            "Set for TEXT, DEFINITION, CALLOUT. A standalone equation belongs in "
            "its own FORMULA block, never inline here — but a short inline math "
            "fragment inside a sentence (e.g. x^n, a_b) must be wrapped in single "
            "$...$ so the frontend can render it. Any inline code (identifier, "
            "statement, command) must be wrapped in single backticks (`code`); "
            "multi-line code belongs in its own CODE block."
        ),
    )
    callout_variant: CalloutVariant | None = Field(default=None, description="Set for CALLOUT only.")
    list_items: list[str] | None = Field(default=None, description="Set for LIST.")
    list_ordered: bool | None = Field(default=None, description="Set for LIST — numbered vs bullet.")
    table: TableData | None = Field(default=None, description="Set for TABLE.")
    formula: FormulaData | None = Field(default=None, description="Set for FORMULA.")
    code_language: str | None = Field(default=None, description="Set for CODE, e.g. 'python'.")
    code: str | None = Field(
        default=None,
        description=(
            "Set for CODE. Real multi-line source: one statement per line, one line "
            "per brace, 4-space indentation, line breaks as actual newlines — never "
            "a function or braced block collapsed on a single line."
        ),
    )
    worked_example: WorkedExample | None = Field(default=None, description="Set for WORKED_EXAMPLE.")
    image_caption: str | None = Field(default=None, description="Set for IMAGE.")
    image_reference: str | None = Field(default=None, description="Set for IMAGE — chunk/page reference of the source figure.")
    pitfall: PitfallData | None = Field(default=None, description="Set for PITFALL — always fill all 3 sub-fields, never leave why_it_happens/how_to_avoid blank.")


class Subsection(BaseModel):
    title: str = Field(
        description=(
            "Subsection heading. Under a DEVELOPMENT section, use exactly "
            "'Quoi', 'Pourquoi', 'Comment' — all three are mandatory, never "
            "omit one — so downstream mapping stays reliable. Other section "
            "types may use free-form titles. Each DEVELOPMENT section covers "
            "one focused sub-topic, not the entire course."
        )
    )
    blocks: list[ContentBlock] = Field(description="Ordered content blocks for this subsection.")


class Section(BaseModel):
    type: SectionType = Field(description="Structural role of the section.")
    title: str = Field(description="Section heading shown to the learner.")
    blocks: list[ContentBlock] = Field(default_factory=list, description="Content directly in the section (no subsection needed).")
    subsections: list[Subsection] = Field(default_factory=list, description="Subsections, e.g. Quoi/Pourquoi/Comment under 'development'.")
    covered_subtopics: list[str] = Field(
        default_factory=list,
        description=(
            "Only when generating from a validated plan: the planned subtopics this "
            "section really explains, each copied VERBATIM from the plan. List a "
            "subtopic only if the text above develops it (definition/mechanism and "
            "concrete items), never if it is merely mentioned."
        ),
    )


class QuizDifficulty(str, Enum):
    FACILE = "facile"
    NORMALE = "normale"
    DIFFICILE = "difficile"


class QuizQuestion(BaseModel):
    question: str
    choices: list[str] = Field(description="Answer options, 2-5 items.")
    correct_indices: list[int] = Field(
        description=(
            "0-based indices into `choices`. Exactly one element for a single-"
            "answer question, multiple elements for a multi-answer (QCM) question."
        ),
    )
    difficulty: QuizDifficulty = Field(
        description=(
            "Difficulty level of this question. Counts must follow the rule: "
            "difficile = round(N/2), normale = round(N/4), "
            "facile = N - difficile - normale."
        ),
    )
    explanation: str = Field(description="Why the correct answer(s) is/are correct.")
    requires_calculation: bool = Field(
        description=(
            "True if answering requires performing a calculation, not just "
            "recalling a definition. Drives the timer downstream (80s vs 45s) — "
            "don't compute the timer value yourself, just flag this."
        )
    )

    @model_validator(mode="after")
    def _check_correct_indices(self) -> "QuizQuestion":
        if not self.correct_indices:
            raise ValueError("correct_indices ne peut pas être vide")
        if len(self.correct_indices) != len(set(self.correct_indices)):
            raise ValueError("correct_indices contient des doublons")
        for idx in self.correct_indices:
            if idx < 0 or idx >= len(self.choices):
                raise ValueError(
                    f"correct_index {idx} hors bornes "
                    f"(choices a {len(self.choices)} éléments, index 0..{len(self.choices)-1})"
                )
        return self


class CoverageCompletionSchema(BaseModel):
    """Schéma léger pour l'appel Gemini de complétion de couverture.

    Réutilise Section existante — pas de nouveau model_validator nécessaire.
    """

    sections: list[Section] = Field(
        description=(
            "Nouvelles sections thématiques (ou sections destinées à enrichir "
            "une section existante de même sujet) couvrant le contenu fourni. "
            "Chaque section suit Quoi/Pourquoi/Comment comme le reste du cours. "
            "INTERDIT : tout titre générique du type 'Contenu complémentaire', "
            "'Pages non couvertes', 'Supplément' — donner un titre thématique réel."
        )
    )


class SectionsBatchSchema(BaseModel):
    """Schéma Gemini pour un lot de sections DEVELOPMENT générées à partir d'un plan validé."""

    sections: list[Section] = Field(
        description=(
            "Exactement une section DEVELOPMENT par section planifiée du lot, dans "
            "le même ordre et avec le même titre que dans le plan. Chaque section "
            "contient les sous-sections Quoi / Pourquoi / Comment (toutes "
            "obligatoires) et un exemple travaillé complet dans Comment."
        )
    )


class PlanMeta(BaseModel):
    title: str = Field(description="Titre du cours.")
    subject: str = Field(description="Matière / domaine du cours.")
    language: str = Field(default="fr")


class PlannedSection(BaseModel):
    """Section planifiée : structure uniquement, aucun contenu Quoi/Pourquoi/Comment rédigé."""

    type: SectionType = Field(description="Rôle structurel de la section dans le cours.")
    title: str = Field(description="Titre thématique réel et précis (jamais générique).")
    objective: str = Field(description="Ce que l'apprenant doit savoir/savoir faire à l'issue de la section.")
    subtopics: list[str] = Field(
        default_factory=list,
        description="Notions, mécanismes ou cas précis que la section devra développer.",
    )
    order: int = Field(description="Position 1-based dans le cours, respectant l'ordre de dépendance logique.")


class CoursePlanSchema(BaseModel):
    """Sortie structurée Gemini de l'étape de planification (structure du cours, sans contenu rédigé)."""

    meta: PlanMeta
    planned_sections: list[PlannedSection] = Field(
        description=(
            "Plan détaillé et complet, ordonné par dépendances logiques "
            "(introduction → notions de base → notions dépendantes → pièges → "
            "résumé → suite). Aucun plafond de sections : la couverture "
            "exhaustive du sujet prime."
        )
    )
    coverage_notes: str = Field(
        default="",
        description="Remarques sur la couverture : ce qui est volontairement exclu ou non confirmé par les sources.",
    )

    @model_validator(mode="after")
    def _check_has_development_section(self) -> "CoursePlanSchema":
        if not any(s.type is SectionType.DEVELOPMENT for s in self.planned_sections):
            raise ValueError("Le plan doit contenir au moins une section 'development'")
        return self


class MoreSectionsSchema(BaseModel):
    """Sortie structurée Gemini : nouvelles sections de développement d'un plan existant."""

    planned_sections: list[PlannedSection] = Field(
        description=(
            "Nouvelles sections DEVELOPMENT (structure uniquement), distinctes des sections "
            "déjà présentes dans le plan, qui développent les pistes « pour aller plus loin »."
        )
    )
    next_steps: PlannedSection | None = Field(
        default=None,
        description=(
            "Section « Pour aller plus loin » (type next_steps) mise à jour : 3 à 5 NOUVELLES pistes qui "
            "prolongent le plan APRÈS ces nouvelles sections. Ne reprend aucune piste déjà développée "
            "(celles qui viennent de devenir des sections) ni aucun sujet déjà présent dans le plan."
        ),
    )


class Meta(BaseModel):
    title: str
    subject: str
    language: str = Field(default="fr")
    generated_at: datetime


class CourseGenerationSchema(BaseModel):
    """Unified structured output for Mode 2 and Mode 3.

    Content lives entirely in `sections` (block-based), so `focused_answer`
    and any future `full_course` format share the same rendering model —
    no more parallel flat `answer` object.
    """

    mode: InteractionMode
    format: OutputFormat
    meta: Meta
    sources: list[Source] = Field(default_factory=list)

    sections: list[Section] = Field(
        description=(
            "Break the content into MULTIPLE DEVELOPMENT sections — one per "
            "logical topic or sub-concept. Each DEVELOPMENT section must have "
            "Quoi/Pourquoi/Comment subsections. Example for a course on "
            "regression: Section 'Introduction', Section 'Le modèle', "
            "Section 'Estimateur', Section 'Métriques d'évaluation'. "
            "Optionally add INTRODUCTION, COMMON_PITFALLS, SUMMARY, NEXT_STEPS."
        )
    )
    quiz: list[QuizQuestion] = Field(
        default_factory=list,
        description=(
            "Empty when not relevant to the current mode. Question count is "
            "driven by how much content was actually covered — no fixed number."
        ),
    )

    video_suggestions: list[VideoSuggestion] = Field(
        default_factory=list,
        description=(
            "1 to 3 YouTube videos that explain the course topic well (preferably in French, from "
            "reputable educational channels). ONLY real videos you found through search or are certain "
            "exist — never invent a URL: each URL is checked and dropped if the video does not exist."
        ),
    )

    confidence: ConfidenceLevel
    unconfirmed_points: list[str] = Field(
        default_factory=list,
        description="Facts that could not be confirmed by context or search — state explicitly instead of inventing.",
    )

    @model_validator(mode="after")
    def _check_format_matches_mode(self) -> "CourseGenerationSchema":
        """FILE_QUESTION and QUESTION_ONLY are Mode 2/3 only — full_course is
        reserved for a future file-only trigger that doesn't call Gemini yet.
        Rejecting the mismatch here means it's caught by the existing
        model_validate() try/except in course_generator.py, instead of
        surfacing 90 lines later as an unrelated ValidationError on
        CourseGenerationResponse.
        """
        if self.mode in (InteractionMode.FILE_QUESTION, InteractionMode.QUESTION_ONLY) and self.format is not OutputFormat.FOCUSED_ANSWER:
            raise ValueError(
                f"mode='{self.mode.value}' requires format='focused_answer', got '{self.format.value}'"
            )
        return self

    @model_validator(mode="after")
    def _check_quiz_difficulty_distribution(self) -> "CourseGenerationSchema":
        """Vérifie la répartition de difficulté dans le quiz.

        Distribution attendue : ~50% difficile, ~25% normale, ~25% facile.
        Tolérance : ±1 question par catégorie (arrondi pour N non multiple de 4).
        Le quiz vide est autorisé (pas de quiz si pas de contenu pertinent).
        """
        if not self.quiz or len(self.quiz) < 2:
            return self
        n = len(self.quiz)
        counts = {d: 0 for d in QuizDifficulty}
        for q in self.quiz:
            counts[q.difficulty] += 1
        expected_difficile = n / 2
        expected_normale = n / 4
        expected_facile = n / 4
        if abs(counts[QuizDifficulty.DIFFICILE] - expected_difficile) > 1:
            raise ValueError(
                f"Répartition de difficulté incorrecte : {counts[QuizDifficulty.DIFFICILE]} "
                f"difficile(s) pour {n} questions (attendu ≈{expected_difficile:.0f})"
            )
        if abs(counts[QuizDifficulty.NORMALE] - expected_normale) > 1:
            raise ValueError(
                f"Répartition de difficulté incorrecte : {counts[QuizDifficulty.NORMALE]} "
                f"normale(s) pour {n} questions (attendu ≈{expected_normale:.0f})"
            )
        if abs(counts[QuizDifficulty.FACILE] - expected_facile) > 1:
            raise ValueError(
                f"Répartition de difficulté incorrecte : {counts[QuizDifficulty.FACILE]} "
                f"facile(s) pour {n} questions (attendu ≈{expected_facile:.0f})"
            )
        return self