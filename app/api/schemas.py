from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


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


class PDFIngestResponse(BaseModel):
    status: str
    filename: str
    chunks_added: int
    documents_added: int


class DocumentQueryResponse(BaseModel):
    query: str
    results: list[dict]


# --- Génération de cours structuré (Mode 2 : fichier + question) ---------------
# Enveloppe commune aux 3 modes (file_only / file_question / question_only) :
# `mode` et `sources` sont toujours présents pour la traçabilité et le routage
# front-end. `format` détermine la richesse du contenu attendu.


class Step(BaseModel):
    id: str = Field(..., description="Identifiant stable de l'étape, ex. 'step-1' (utile comme clé React côté front)")
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
    question: str
    options: list[str] = Field(..., min_length=2)
    correct_option_index: int = Field(..., ge=0)
    explanation: str


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


class CourseGenerationRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question de l'utilisateur")
    top_k: int = Field(6, ge=1, le=20, description="Nombre de chunks à récupérer pour le contexte")
    filename: str | None = Field(None, description="Filtre optionnel sur un document déjà ingéré")

    @field_validator("question")
    @classmethod
    def strip_question(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question ne peut pas être vide")
        return v


class CourseGenerationResponse(BaseModel):
    mode: Literal["file_only", "file_question", "question_only"]
    format: Literal["full_course", "focused_answer"]
    meta: CourseMeta
    sources: list[CourseSource]

    # Uniquement pour format == "full_course"
    introduction: dict[str, str] | None = None
    sections: list[CourseSection] | None = None
    common_pitfalls: list[CoursePitfall] | None = None
    quiz: list[QuizQuestion] | None = None

    # Uniquement pour format == "focused_answer"
    answer: CourseAnswer | None = None

    summary: str
    next_steps: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_format_consistency(self) -> "CourseGenerationResponse":
        if self.format == "full_course" and (self.introduction is None or self.sections is None):
            raise ValueError("format='full_course' nécessite 'introduction' et 'sections'")
        if self.format == "focused_answer" and self.answer is None:
            raise ValueError("format='focused_answer' nécessite 'answer'")
        return self