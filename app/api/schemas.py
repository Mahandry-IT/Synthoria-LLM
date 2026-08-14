from pydantic import BaseModel, Field, field_validator


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
    top_k: int = Field(5, ge=1, le=10, description="Nombre de résultats vectoriels à renvoyer")


class PDFIngestResponse(BaseModel):
    status: str
    filename: str
    chunks_added: int
    documents_added: int


class DocumentQueryResponse(BaseModel):
    query: str
    results: list[dict]
