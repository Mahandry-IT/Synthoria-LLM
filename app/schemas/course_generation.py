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
    headers: list[str] = Field(description="Column headers, in order.")
    rows: list[list[str]] = Field(description="Row values, each row matching the headers length.")


class FormulaData(BaseModel):
    latex: str = Field(description="Formula in LaTeX, exact — never round or simplify silently.")
    description: str | None = Field(default=None, description="Short plain-language reading of the formula.")


class WorkedExample(BaseModel):
    statement: str = Field(description="Concrete, non-placeholder statement of the example (with real numbers/data).")
    steps: list[str] = Field(description="Explicit intermediate steps, in order. Never skip to the final result.")
    result: str = Field(description="Final result, commented — what it means, not just the raw value.")


class PitfallData(BaseModel):
    description: str = Field(description="The mistake itself — what learners commonly get wrong.")
    why_it_happens: str = Field(description="Root cause: why learners fall into this, not just that they do.")
    how_to_avoid: str = Field(description="Concrete corrective action or check to prevent it.")


class ContentBlock(BaseModel):
    """Single typed content unit. Only the field(s) matching `type` are populated."""

    type: BlockType = Field(description="Kind of content this block carries; determines which field below is set.")

    text: str | None = Field(default=None, description="Set for TEXT, DEFINITION, CALLOUT.")
    callout_variant: CalloutVariant | None = Field(default=None, description="Set for CALLOUT only.")
    list_items: list[str] | None = Field(default=None, description="Set for LIST.")
    list_ordered: bool | None = Field(default=None, description="Set for LIST — numbered vs bullet.")
    table: TableData | None = Field(default=None, description="Set for TABLE.")
    formula: FormulaData | None = Field(default=None, description="Set for FORMULA.")
    code_language: str | None = Field(default=None, description="Set for CODE, e.g. 'python'.")
    code: str | None = Field(default=None, description="Set for CODE.")
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
            "types may use free-form titles."
        )
    )
    blocks: list[ContentBlock] = Field(description="Ordered content blocks for this subsection.")


class Section(BaseModel):
    type: SectionType = Field(description="Structural role of the section.")
    title: str = Field(description="Section heading shown to the learner.")
    blocks: list[ContentBlock] = Field(default_factory=list, description="Content directly in the section (no subsection needed).")
    subsections: list[Subsection] = Field(default_factory=list, description="Subsections, e.g. Quoi/Pourquoi/Comment under 'development'.")


class QuizQuestion(BaseModel):
    question: str
    choices: list[str] = Field(description="Answer options, 2-5 items.")
    correct_index: int = Field(description="0-based index into `choices`.")
    explanation: str = Field(description="Why the correct answer is correct.")


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
            "For focused_answer: typically one DEVELOPMENT section with "
            "Quoi/Pourquoi/Comment subsections, optionally a SUMMARY section. "
            "For full_course: INTRODUCTION, DEVELOPMENT (one subsection per topic), "
            "COMMON_PITFALLS, SUMMARY, NEXT_STEPS."
        )
    )
    quiz: list[QuizQuestion] = Field(default_factory=list, description="Empty when not relevant to the current mode.")

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