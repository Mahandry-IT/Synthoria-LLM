import logging
import math
import re
from datetime import datetime
from typing import Annotated, Any, Generic, Literal, TypeVar, get_args
from uuid import UUID

from fastapi import Query
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from app.core.config import get_settings
from app.db.models import DEFAULT_COURSE_FOLDER, DEFAULT_COURSE_SUBFOLDER
from app.services.course_depth import LEGACY_DEPTH, CourseDepth

T = TypeVar("T")

logger = logging.getLogger(__name__)


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


class GeminiModelHealth(BaseModel):
    available: bool = Field(..., description="False si épuisé (quota jour) ou budget RPD configuré atteint.")
    requests_today: int = Field(..., description="Compteur d'appels réussis aujourd'hui, partagé api/worker.")
    exhausted_until: str | None = Field(None, description="ISO 8601 (UTC) si marqué épuisé, sinon null.")


class HealthResponse(BaseModel):
    status: str
    ollama_reachable: bool
    gemini: dict[str, GeminiModelHealth] | None = Field(
        None, description="État par modèle Gemini configuré ; absent si la DB de quota est indisponible."
    )


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
    folder: str = Field(DEFAULT_COURSE_FOLDER, description="Dossier de rangement")
    subfolder: str = Field(DEFAULT_COURSE_SUBFOLDER, description="Sous-dossier de rangement")


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


class CourseTable(BaseModel):
    """Tableau structuré (rendu graphique côté front, plutôt que du texte aplati)."""
    caption: str = ""
    headers: list[str]
    rows: list[list[str]]


# Limites de rendu : sources non fiables (LLM), on borne pour éviter un rendu bloquant côté client.
DIAGRAM_SOURCE_MAX = 4000
CHART_LABELS_MAX = 12
CHART_SERIES_MAX = 4
CHART_STEPS_MAX = 8
CHART_STEP_CAPTION_MAX = 240


class ApiDiagram(BaseModel):
    kind: Literal["flowchart", "sequence", "hierarchy", "cycle"]
    caption: str = ""
    mermaid: str = Field(..., min_length=1, max_length=DIAGRAM_SOURCE_MAX)


class ApiChartSeries(BaseModel):
    name: str
    values: list[float] = Field(..., max_length=CHART_LABELS_MAX)


class ApiChartStep(BaseModel):
    caption: str = Field(..., min_length=1, max_length=CHART_STEP_CAPTION_MAX)
    highlight_labels: list[int] = Field(default_factory=list)
    highlight_series: list[int] = Field(default_factory=list)


def _clean_indices(raw: object, size: int) -> list[int]:
    """Indices valides (0 <= i < size), sans doublon, ordre conservé ; le reste est ignoré."""
    if not isinstance(raw, list):
        return []
    seen: list[int] = []
    for i in raw:
        if isinstance(i, int) and not isinstance(i, bool) and 0 <= i < size and i not in seen:
            seen.append(i)
    return seen


def _clean_chart_steps(raw: object, n_labels: int, n_series: int) -> list[dict]:
    """Tolérant : étape sans légende retirée, indices hors bornes retirés, tronqué à CHART_STEPS_MAX."""
    if not isinstance(raw, list):
        return []
    steps: list[dict] = []
    for item in raw:
        if hasattr(item, "model_dump"):
            item = item.model_dump()
        if not isinstance(item, dict):
            continue
        caption = item.get("caption")
        caption = caption.strip()[:CHART_STEP_CAPTION_MAX].strip() if isinstance(caption, str) else ""
        if not caption:
            continue
        steps.append({
            "caption": caption,
            "highlight_labels": _clean_indices(item.get("highlight_labels"), n_labels),
            "highlight_series": _clean_indices(item.get("highlight_series"), n_series),
        })
        if len(steps) == CHART_STEPS_MAX:
            break
    return steps


class ApiChart(BaseModel):
    kind: Literal["bar", "line", "pie"]
    caption: str = ""
    labels: list[str] = Field(..., min_length=1, max_length=CHART_LABELS_MAX)
    series: list[ApiChartSeries] = Field(..., min_length=1, max_length=CHART_SERIES_MAX)
    steps: list[ApiChartStep] = Field(default_factory=list, max_length=CHART_STEPS_MAX)

    @model_validator(mode="before")
    @classmethod
    def _sanitize_steps(cls, data: object) -> object:
        # Appliqué sur tous les chemins (génération, relecture) : une étape invalide ne fait jamais échouer le graphique.
        if isinstance(data, dict) and "steps" in data:
            labels, series = data.get("labels"), data.get("series")
            data = {**data, "steps": _clean_chart_steps(
                data["steps"],
                len(labels) if isinstance(labels, list) else 0,
                len(series) if isinstance(series, list) else 0,
            )}
        return data


class ApiFormula(BaseModel):
    latex: str
    description: str | None = None


class ApiWorkedExample(BaseModel):
    statement: str = ""
    steps: list[str] = Field(default_factory=list)
    result: str = ""


class ApiPitfall(BaseModel):
    description: str
    why_it_happens: str = ""
    how_to_avoid: str = ""


class ApiImageAttribution(BaseModel):
    """Attribution obligatoire pour les licences CC BY / CC BY-SA (jamais affiché sans elle)."""

    author: str | None = None
    license: str | None = None
    license_url: str | None = None


class ApiImage(BaseModel):
    """Image ré-hébergée, jamais de hotlink : `url` pointe toujours vers GET /media/{asset_id}."""

    asset_id: str
    url: str
    alt: str = ""
    caption: str = ""
    width: int
    height: int
    attribution: ApiImageAttribution | None = None


WIDGET_TITLE_MAX = 120
WIDGET_DESCRIPTION_MAX = 500
WIDGET_HEIGHT_MIN = 160
WIDGET_HEIGHT_MAX = 720

# Espaces de noms XML standards : indispensables à `createElementNS` / `xmlns` d'un SVG construit en JS,
# sans effet réseau. Retirés avant la recherche d'URL.
_XML_NAMESPACES_RE = re.compile(r"http://www\.w3\.org/(?:2000/svg|1999/xlink|1999/xhtml)", re.I)
# (libellé, motif) : défense en profondeur ; le vrai rempart est l'iframe sandboxée + CSP côté client.
_WIDGET_FORBIDDEN: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (label, re.compile(pattern, re.I))
    for label, pattern in (
        ("balise interdite", r"<\s*/?\s*(?:html|head|body|meta|base|iframe|object|embed|link|form)\b"),
        ("script externe", r"<\s*script\b[^>]*\bsrc\b"),
        ("URL absolue", r"https?://"),
        ("URL relative au protocole", r"""["'(=]\s*//[\w.\-]"""),
        ("@import", r"@import"),
        ("requête réseau", r"\bfetch\s*\(|XMLHttpRequest|WebSocket|EventSource|sendBeacon|\bimport\s*\("),
        ("accès au parent", r"\b(?:window|self|globalThis)\s*\.\s*(?:parent|top)\b|(?<![\w$.\-])(?:parent|top)\.[A-Za-z_$]"),
        ("stockage navigateur", r"document\s*\.\s*cookie|localStorage|sessionStorage|indexedDB"),
        ("exécution dynamique", r"\beval\s*\(|\bnew\s+Function\b|javascript\s*:"),
        ("postMessage / srcdoc", r"postMessage|srcdoc"),
    )
)


def widget_html_problem(html: str) -> str | None:
    """Libellé du premier motif interdit trouvé dans `html` (None si le fragment est acceptable)."""
    if len(html) > get_settings().course_widget_max_chars:
        return "taille maximale dépassée"
    scanned = _XML_NAMESPACES_RE.sub("", html)
    for label, pattern in _WIDGET_FORBIDDEN:
        if pattern.search(scanned):
            return label
    return None


class ApiWidget(BaseModel):
    """Visuel interactif autonome (fragment de body : HTML, `<style>`, `<script>` inline), affiché par le
    client dans une iframe sandboxée. `description` est le texte alternatif, aussi lu par le chatbot et
    le podcast. Le filtre de `html` s'applique partout où le modèle est validé."""

    title: str = Field(min_length=1, max_length=WIDGET_TITLE_MAX)
    description: str = Field(min_length=1, max_length=WIDGET_DESCRIPTION_MAX)
    html: str = Field(min_length=1)
    height_hint: int | None = Field(default=None, ge=WIDGET_HEIGHT_MIN, le=WIDGET_HEIGHT_MAX)

    @field_validator("html")
    @classmethod
    def _html_is_safe(cls, value: str) -> str:
        problem = widget_html_problem(value)
        if problem:
            # Jamais le HTML dans le message : il finirait dans les logs et les réponses d'erreur.
            raise ValueError(f"widget html refusé : {problem}")
        return value


class ApiContentBlock(BaseModel):
    """Bloc de contenu typé (contrat par blocs). Seuls les champs correspondant à `type` sont renseignés."""

    type: Literal[
        "text", "definition", "list", "table", "formula", "code", "worked_example",
        "callout", "image", "pitfall", "diagram", "chart", "widget",
    ]
    text: str | None = None
    callout_variant: Literal["note", "warning", "tip"] | None = None
    list_items: list[str] | None = None
    list_ordered: bool | None = None
    table: CourseTable | None = None
    formula: ApiFormula | None = None
    code_language: str | None = None
    code: str | None = None
    worked_example: ApiWorkedExample | None = None
    image_caption: str | None = None
    image: ApiImage | None = None
    pitfall: ApiPitfall | None = None
    diagram: ApiDiagram | None = None
    chart: ApiChart | None = None
    widget: ApiWidget | None = None

    # Intention Gemini portée jusqu'au résolveur (app/services/media/visual_resolver.py), jamais
    # exposée aux clients de l'API : ni URL ni identifiant externe, juste de quoi résoudre le
    # visuel serveur (cf. plan-supports-visuels.md, principe « Gemini n'émet jamais d'URL »).
    image_source: str | None = Field(default=None, exclude=True)
    image_query: str | None = Field(default=None, exclude=True)
    image_reference: str | None = Field(default=None, exclude=True)
    image_alt: str | None = Field(default=None, exclude=True)


class CourseSubsection(BaseModel):
    title: str = ""
    blocks: list[ApiContentBlock] = Field(default_factory=list)

    @field_validator("blocks", mode="before")
    @classmethod
    def _drop_unknown_block_types(cls, value: Any) -> Any:
        """Les cours enregistrés (JSON brut) peuvent contenir des blocs d'un type retiré depuis (ex. `graph`) :
        on les ignore à la lecture plutôt que de faire échouer tout le cours."""
        if not isinstance(value, list):
            return value
        known = set(get_args(ApiContentBlock.model_fields["type"].annotation))
        return [b for b in value if not (isinstance(b, dict) and b.get("type") not in known)]


class CourseVideo(BaseModel):
    """Vidéo YouTube vérifiée, affichée en carte au-dessus du podcast."""
    video_id: str
    url: str
    embed_url: str
    thumbnail_url: str
    title: str
    channel: str = ""
    duration_seconds: int | None = Field(None, description="Durée de la vidéo, en secondes (YouTube Data API).")
    published_at: str | None = Field(None, description="Date de publication ISO 8601 (YouTube Data API).")
    category: str | None = Field(None, description="Catégorie pédagogique (V2 : classement Gemini).")
    level: str | None = Field(None, description="Niveau estimé (V2 : classement Gemini).")
    relevance_reason: str | None = Field(None, description="Pourquoi cette vidéo a été retenue (V2).")
    section_id: str | None = Field(
        None,
        description=(
            "`id` de la section que cette vidéo illustre le mieux (à afficher dans cette section) ; "
            "null = vidéo générale du cours, ou section inconnue/supprimée."
        ),
    )
    note: str = Field(
        "", description="Note libre de l'apprenant sur cette vidéo (pense-bête, idées) ; jamais générée par le modèle."
    )


class ApiFadedExample(BaseModel):
    statement: str = ""
    given_steps: list[str] = Field(default_factory=list)
    hidden_steps: list[str] = Field(default_factory=list)
    result: str = ""


class ApiRecallPrompt(BaseModel):
    prompt: str
    expected_key_points: list[str] = Field(default_factory=list)


class ApiPitfallToAvoid(BaseModel):
    """Piège d'un encadré « Ce qu'il faut éviter » : texte du modèle, à afficher sans HTML."""

    mistake: str = Field("", max_length=300)
    how_to_avoid: str = Field("", max_length=300)


class CourseSection(BaseModel):
    id: str
    title: str
    quoi: str
    pourquoi: str
    comment: str
    worked_example: WorkedExample
    key_points: list[str] = Field(default_factory=list)
    tables: list[CourseTable] = Field(default_factory=list, description="Tableaux de la section (hors texte quoi/pourquoi/comment).")
    challenge: str = Field("", description="Défi posé avant l'explication (cycle pédagogique).")
    challenge_key_points: list[str] = Field(
        default_factory=list,
        description=(
            "2-4 idées attendues dans une réponse au défi (évaluation côté serveur) ; vide pour les "
            "anciennes sessions (repli sur Pourquoi/Quoi)."
        ),
    )
    key_takeaways: list[Annotated[str, Field(max_length=200)]] = Field(
        default_factory=list, max_length=6,
        description="2-4 puces « Ce qu'il faut retenir » ; vide pour les cours antérieurs au format.",
    )
    pitfalls_to_avoid: list[ApiPitfallToAvoid] = Field(
        default_factory=list, max_length=3,
        description="1-2 pièges « Ce qu'il faut éviter » ; vide pour les cours antérieurs au format.",
    )
    faded_example: "ApiFadedExample | None" = None
    check_questions: list["QuizQuestion"] = Field(default_factory=list, description="2-3 questions « Vérifie ».")
    recall_prompt: "ApiRecallPrompt | None" = Field(
        None, description="Consigne « explique avec tes mots » ; les points attendus servent à l'évaluation côté serveur."
    )
    subsections: list[CourseSubsection] = Field(
        default_factory=list,
        description=(
            "Contenu par blocs typés (référence). `quoi/pourquoi/comment/worked_example/tables` sont "
            "dépréciés : conservés pour l'historique et le podcast pendant une version."
        ),
    )
    cycle: "ApiSectionCycle | None" = Field(
        None,
        description=(
            "Format « cycle » (Défi, Sous le capot, Le mot juste, Les pièges classiques, Épreuve) des sections de "
            "développement des cours en mode standard ; absent = ancien format (Pourquoi / Quoi / Comment), "
            "toujours valide. Les options des QCM et des affirmations sont déjà mélangées par le serveur. Dans ce "
            "format, `quoi/pourquoi/comment` ne portent qu'un texte à plat (podcast, assistant, détection)."
        ),
    )
    widget_brief: "ApiWidgetBrief | None" = Field(
        None, description="Visuel simulable demandé à la section (usage serveur) ; absent hors format cycle."
    )
    incomplete: bool = Field(
        False,
        description="Génération de cette section échouée (erreur temporaire) : seule une section incomplète peut être régénérée.",
    )
    note: str = Field(
        "", description="Note libre de l'apprenant sur cette section (pense-bête, idées) ; jamais générée par le modèle."
    )

    @field_validator("cycle", "widget_brief", mode="wrap")
    @classmethod
    def _tolerant_cycle(cls, value: Any, handler: Any) -> Any:
        """Un `cycle` (ou son `widget_brief`) invalide ne fait jamais échouer le cours à la lecture : il est
        ignoré, la section se lit alors comme une section sans cycle (jamais de 500)."""
        try:
            return handler(value)
        except ValidationError:
            logger.warning("course_cycle_dropped", extra={"reason": "validation"})
            return None


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
    kind: Literal["choice", "numeric", "matching"] = Field(
        default="choice", description="Type de question ; absent dans les anciens quiz = choice."
    )
    options: list[str] = Field(
        default_factory=list,
        description="choice : options (>= 2). numeric : []. matching : options à associer, autant que de sujets.",
    )
    expected_value: float | None = Field(default=None, description="numeric : valeur attendue (null avant correction).")
    unit: str | None = Field(default=None, description="numeric : unité facultative.")
    subjects: list[str] = Field(default_factory=list, description="matching : sujets à associer aux options.")
    correct_matches: list[int] = Field(
        default_factory=list,
        description="matching : indice d'option attendu par sujet (vide avant correction).",
    )
    correct_option_indices: list[int] = Field(default=[], min_length=0, description="Indices 0-based des bonnes réponses (1 = unique, >1 = QCM multiple)")
    difficulty: Literal["facile", "normale", "difficile"] = "normale"
    points: float = Field(default=1.0, ge=0.0, description="Points alloués à cette question (calculé côté serveur, borne sup ~2.0 pour N≥10)")
    explanation: str = ""
    explanation_per_choice: list[str] = Field(default_factory=list, description="Retour par option (bonne ou distracteur).")
    section_refs: list[Annotated[int, Field(ge=1, le=500)]] = Field(
        default_factory=list, max_length=10, description="Sections (position 1-based) mobilisées par la question."
    )
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

    @model_validator(mode="after")
    def _check_kind(self) -> "QuizQuestion":
        """Cohérence par type, tolérante à la lecture (les réponses sont vides côté public)."""
        if self.kind == "choice" and len(self.options) < 2:
            raise ValueError("une question choice demande au moins 2 options")
        if self.kind == "numeric" and self.expected_value is not None and not math.isfinite(self.expected_value):
            raise ValueError("expected_value doit être fini")
        if self.kind == "matching":
            n = len(self.subjects)
            if n != len(self.options) or n < 3:
                raise ValueError("matching : autant de sujets que d'options, au moins 3")
            if self.correct_matches and sorted(self.correct_matches) != list(range(n)):
                raise ValueError("correct_matches doit être une permutation des indices d'options")
        return self

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


# --- Format « cycle » d'une section (Défi, Sous le capot, Le mot juste, Les pièges classiques, Épreuve) ---
# Textes du modèle, donc non fiables : bornés ici et affichés sans HTML par le client. Le mapping
# (`app.services.cycle_mapping`) tronque et écarte les éléments invalides AVANT cette validation ; elle reste la
# dernière barrière. Les bonnes réponses font partie du contenu du cours (déjà lisible côté client) : rien de
# secret ici, mais jamais de donnée personnelle.

CYCLE_THREAD_MAX = 200
CYCLE_QUESTION_MAX = 400
CYCLE_OPTION_MAX = 160
CYCLE_FEEDBACK_MAX = 300
CYCLE_STEP_TEXT_MAX = 500
CYCLE_ACTION_MAX = 200
CYCLE_CAPTION_MAX = 120
CYCLE_CELL_MAX = 120
CYCLE_LABEL_MAX = 100
CYCLE_CATEGORY_MAX = 60
CYCLE_ITEM_MAX = 120
CYCLE_OUTCOME_MAX = 300
CYCLE_UNIT_MAX = 20
CYCLE_TERM_MAX = 80
CYCLE_PLAIN_MAX = 300
CYCLE_FORMULA_MAX = 200
CYCLE_CLAIM_MAX = 160
CYCLE_EXPLANATION_MAX = 400
CYCLE_CARD_STATEMENT_MAX = 700
CYCLE_CARD_LINE_MAX = 300
WIDGET_BRIEF_MAX = 300
CYCLE_STEPS_MAX = 8
CYCLE_TABLE_ROWS_MAX = 8
CYCLE_TABLE_COLS_MAX = 6
CYCLE_OPTIONS_MAX = 6
CYCLE_ITEMS_MAX = 8
CYCLE_VOCABULARY_MAX = 10
CYCLE_TRAP_ENTRIES_MAX = 4
CYCLE_TRIAL_QUESTIONS_MAX = 3
CYCLE_TRIAL_OPTIONS_MAX = 5
CYCLE_CARD_STEPS_MAX = 6


def _text(limit: int) -> Any:
    """Texte non vide d'au plus `limit` caractères."""
    return Annotated[str, Field(min_length=1, max_length=limit)]


def _opt_text(limit: int) -> Any:
    return Annotated[str, Field(max_length=limit)]


class ApiCellRef(BaseModel):
    row: int = Field(..., ge=0, le=CYCLE_TABLE_ROWS_MAX)
    col: int = Field(..., ge=0, le=CYCLE_TABLE_COLS_MAX)


class ApiClassifyCase(BaseModel):
    label: _text(CYCLE_LABEL_MAX)
    category_index: int = Field(..., ge=0, le=CYCLE_OPTIONS_MAX)


class ApiMatchPair(BaseModel):
    left: _text(CYCLE_LABEL_MAX)
    right: _text(CYCLE_LABEL_MAX)


class ApiCycleChallenge(BaseModel):
    """Défi posé avant toute explication : cases à cocher, sans annoncer le nombre de bonnes réponses.
    L'ordre des options est déjà mélangé par le serveur ; `correct_indices` suit cet ordre."""

    question: _text(CYCLE_QUESTION_MAX)
    options: list[_text(CYCLE_OPTION_MAX)] = Field(..., min_length=2, max_length=CYCLE_OPTIONS_MAX)
    correct_indices: list[int] = Field(..., min_length=1)
    feedback: _opt_text(CYCLE_FEEDBACK_MAX) = ""

    @model_validator(mode="after")
    def _check_correct_indices(self) -> "ApiCycleChallenge":
        """Comme une prédiction `choice` : index dans les options, sans doublon, jamais toutes les options."""
        indices = self.correct_indices
        if any(not 0 <= i < len(self.options) for i in indices):
            raise ValueError("défi : correct_indices hors des options")
        if len(set(indices)) != len(indices):
            raise ValueError("défi : correct_indices en doublon")
        if len(indices) >= len(self.options):
            raise ValueError("défi : correct_indices ne doit jamais désigner toutes les options")
        return self


class ApiStepSupport(BaseModel):
    """Support léger d'une mini-étape ; seuls les champs du groupe de `kind` sont renseignés."""

    kind: Literal["chart", "diagram", "table", "fill_table", "classify", "order", "match"]
    caption: _opt_text(CYCLE_CAPTION_MAX) = ""
    chart: ApiChart | None = None
    diagram: ApiDiagram | None = None
    table: CourseTable | None = None
    blanks: list[ApiCellRef] = Field(default_factory=list, max_length=4)
    distractors: list[_text(CYCLE_CELL_MAX)] = Field(default_factory=list, max_length=3)
    categories: list[_text(CYCLE_CATEGORY_MAX)] = Field(default_factory=list, max_length=4)
    cases: list[ApiClassifyCase] = Field(default_factory=list, max_length=CYCLE_ITEMS_MAX)
    order_items: list[_text(CYCLE_ITEM_MAX)] = Field(default_factory=list, max_length=CYCLE_ITEMS_MAX)
    order_criterion: _opt_text(CYCLE_ITEM_MAX) = ""
    match_pairs: list[ApiMatchPair] = Field(default_factory=list, max_length=CYCLE_ITEMS_MAX)


class ApiStepPrediction(BaseModel):
    """Prédiction posée avant la manipulation (jamais notée). `choice` : options déjà mélangées par le
    serveur, `correct_index` suit cet ordre. `value` : `tolerance` absolue, bornée à 10 % de la valeur."""

    kind: Literal["choice", "value"]
    question: _text(CYCLE_QUESTION_MAX)
    options: list[_text(CYCLE_OPTION_MAX)] = Field(default_factory=list, max_length=4)
    correct_index: int | None = Field(default=None, ge=0, le=3)
    expected_value: float | None = None
    tolerance: float | None = Field(default=None, ge=0)
    unit: _opt_text(CYCLE_UNIT_MAX) | None = None
    outcome: _text(CYCLE_OUTCOME_MAX)

    @model_validator(mode="after")
    def _check_kind(self) -> "ApiStepPrediction":
        if self.kind == "choice":
            if len(self.options) < 2 or self.correct_index is None or self.correct_index >= len(self.options):
                raise ValueError("prédiction choice : 2 à 4 options et un correct_index valide")
        elif self.expected_value is None or not math.isfinite(self.expected_value):
            raise ValueError("prédiction value : expected_value fini obligatoire")
        return self


class ApiCycleStep(BaseModel):
    """Mini-étape de « Sous le capot ». `widget` : visuel simulable de la section, attaché à l'étape que
    `widget_brief` désigne (au plus une étape par section) ; absent si la génération n'en a pas produit."""

    text: _text(CYCLE_STEP_TEXT_MAX)
    action: _opt_text(CYCLE_ACTION_MAX) = ""
    support: ApiStepSupport
    prediction: ApiStepPrediction | None = None
    widget: ApiWidget | None = None

    @field_validator("widget", mode="wrap")
    @classmethod
    def _tolerant_widget(cls, value: Any, handler: Any) -> Any:
        """Un widget invalide (HTML refusé à la relecture, par exemple) est ignoré : il ne doit pas faire
        tomber tout le cycle, le repli de reconstruction de l'étape suffit."""
        try:
            return handler(value)
        except ValidationError:
            logger.warning("course_cycle_widget_dropped", extra={"reason": "validation"})
            return None


class ApiVocabTerm(BaseModel):
    term: _text(CYCLE_TERM_MAX)
    plain: _text(CYCLE_PLAIN_MAX)
    formula_latex: _opt_text(CYCLE_FORMULA_MAX) | None = None


class ApiTrapEntry(BaseModel):
    """Piège classique. `cards` : `claim` + `explanation`. `spot` : `statements` (une seule fausse, `wrong_index`,
    ordre déjà mélangé par le serveur). `spot_rewrite` : en plus `rewrite_options` / `correct_rewrite_index`."""

    claim: _opt_text(CYCLE_CLAIM_MAX) = ""
    statements: list[_text(CYCLE_CLAIM_MAX)] = Field(default_factory=list, max_length=5)
    wrong_index: int | None = Field(default=None, ge=0, le=4)
    explanation: _text(CYCLE_EXPLANATION_MAX)
    rewrite_options: list[_text(CYCLE_CLAIM_MAX)] = Field(default_factory=list, max_length=4)
    correct_rewrite_index: int | None = Field(default=None, ge=0, le=3)


class ApiCycleTraps(BaseModel):
    mode: Literal["cards", "spot", "spot_rewrite"]
    entries: list[ApiTrapEntry] = Field(default_factory=list, max_length=CYCLE_TRAP_ENTRIES_MAX)


class ApiExerciseCard(BaseModel):
    """Carte-exercice : énoncé, indice sur demande, correction pas à pas après la tentative. Pas de valeur attendue."""

    statement: _text(CYCLE_CARD_STATEMENT_MAX)
    hint: _opt_text(CYCLE_CARD_LINE_MAX) = ""
    steps: list[_text(CYCLE_CARD_LINE_MAX)] = Field(..., min_length=1, max_length=CYCLE_CARD_STEPS_MAX)
    result: _opt_text(CYCLE_CARD_LINE_MAX) = ""


class ApiTrialQuestion(QuizQuestion):
    """Question de l'épreuve : une `QuizQuestion` (choix ou valeur numérique, jamais d'association) dont le texte
    du modèle est borné, comme tout le reste du cycle. Se sérialise comme une `QuizQuestion`."""

    kind: Literal["choice", "numeric"] = "choice"

    @model_validator(mode="after")
    def _check_bounds(self) -> "ApiTrialQuestion":
        texts = {
            "question": (self.question, CYCLE_QUESTION_MAX),
            "explanation": (self.explanation, CYCLE_EXPLANATION_MAX),
            **{f"options[{i}]": (o, CYCLE_OPTION_MAX) for i, o in enumerate(self.options)},
            **{f"explanation_per_choice[{i}]": (e, CYCLE_EXPLANATION_MAX) for i, e in enumerate(self.explanation_per_choice)},
            "unit": (self.unit or "", CYCLE_UNIT_MAX),
        }
        too_long = [name for name, (text, limit) in texts.items() if len(text) > limit]
        if too_long:
            raise ValueError(f"épreuve : texte trop long ({', '.join(too_long)})")
        if not self.question.strip():
            raise ValueError("épreuve : question vide")
        if len(self.options) > CYCLE_TRIAL_OPTIONS_MAX or len(self.explanation_per_choice) > CYCLE_TRIAL_OPTIONS_MAX:
            raise ValueError(f"épreuve : plus de {CYCLE_TRIAL_OPTIONS_MAX} options")
        return self


class ApiCycleTrial(BaseModel):
    kind: Literal["qcm", "exercise_card"]
    questions: list[ApiTrialQuestion] = Field(default_factory=list, max_length=CYCLE_TRIAL_QUESTIONS_MAX)
    card: ApiExerciseCard | None = None


class ApiWidgetBrief(BaseModel):
    """Intention du visuel simulable de la section (consommée côté serveur par la génération des widgets)."""

    step_number: int = Field(..., ge=1, le=CYCLE_STEPS_MAX)
    brief: _text(WIDGET_BRIEF_MAX)


class ApiSectionCycle(BaseModel):
    """Les cinq temps d'une section de développement, dans l'ordre : `challenge` (Défi), `steps` (Sous le capot),
    `vocabulary` (Le mot juste), `traps` (Les pièges classiques), `trial` (Épreuve). Les titres sont constants
    côté client. Absent des anciens cours, de l'introduction et du résumé."""

    thread: _opt_text(CYCLE_THREAD_MAX) = ""
    challenge: ApiCycleChallenge
    steps: list[ApiCycleStep] = Field(..., min_length=1, max_length=CYCLE_STEPS_MAX)
    vocabulary: list[ApiVocabTerm] = Field(default_factory=list, max_length=CYCLE_VOCABULARY_MAX)
    traps: ApiCycleTraps | None = None
    trial: ApiCycleTrial | None = None
    minutes_min: int | None = Field(
        None, ge=1, le=30,
        description="Durée estimée minimale en minutes, calculée côté serveur ; absente des anciennes sections.",
    )
    minutes_max: int | None = Field(
        None, ge=1, le=30, description="Durée estimée maximale en minutes (>= `minutes_min`), calculée côté serveur."
    )


CourseSection.model_rebuild()


class CourseMeta(BaseModel):
    title: str
    subject: str
    language: str = "fr"
    generated_at: str
    depth: CourseDepth = Field(
        LEGACY_DEPTH, description="Mode du cours (express/standard/approfondi) ; absent des anciens cours = approfondi."
    )


class CourseAnswer(BaseModel):
    """Réponse directe. `quoi/pourquoi/comment/worked_example` : ancien format (historique JSONB)."""

    summary: str = ""
    blocks: list[dict[str, Any]] = Field(
        default_factory=list, description="Blocs de contenu de la réponse (visuel récapitulatif)."
    )
    quoi: str | None = None
    pourquoi: str | None = None
    comment: str | None = None
    worked_example: WorkedExample | None = None
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
    depth: CourseDepth = Field(
        default_factory=lambda: get_settings().default_course_depth,
        description=(
            "Niveau de détail du cours : express (3-5 sections courtes), standard (6-8 sections) ou "
            "approfondi (couverture complète). Défaut : DEFAULT_COURSE_DEPTH (approfondi)."
        ),
    )


class Flashcard(BaseModel):
    """Carte de révision dérivée d'une question « Vérifie » (recto/verso)."""

    card_id: str
    front: str
    back: str
    section_ref: str | None = None


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
    videos: list[CourseVideo] = Field(default_factory=list, description="Vidéos YouTube vérifiées expliquant le cours.")
    flashcards: list[Flashcard] = Field(
        default_factory=list, description="Flashcards dérivées des questions « Vérifie » (répétition espacée)."
    )

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
_PLAN_SUBTOPICS_MAX_ITEMS = 8

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
    mastery: Literal["known"] | None = Field(
        None, description="« known » : l'apprenant maîtrise déjà cette section (pré-test réussi) → version condensée."
    )

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


class ApiPretestItem(BaseModel):
    section_title: str
    question: QuizQuestion


class CoursePlanResponse(BaseModel):
    plan_id: UUID
    expires_at: str
    mode: Literal["file_question", "question_only"]
    depth: CourseDepth = Field(LEGACY_DEPTH, description="Mode du cours choisi à la planification.")
    meta: CoursePlanMeta
    sections: list[ApiPlannedSection]
    pretest: list[ApiPretestItem] = Field(default_factory=list, description="Pré-test diagnostique (1 question par section).")
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


class RecallRequest(BaseModel):
    answer: str = Field(..., min_length=1, max_length=1000, description="Reformulation de l'apprenant (≤ 1000 caractères).")

    @field_validator("answer")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("answer ne peut pas être vide")
        return v


class RecallResponse(BaseModel):
    verdict: Literal["correct", "partiel", "incorrect"]
    feedback: str
    missing_points: list[str] = Field(default_factory=list)


class ChallengeRequest(BaseModel):
    answer: str = Field(
        ..., min_length=1, max_length=1000, description="Réponse de l'apprenant au défi (≤ 1000 caractères)."
    )

    @field_validator("answer")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("answer ne peut pas être vide")
        return v


class ChallengeResponse(BaseModel):
    """Retour sur la réponse au défi : oriente vers l'explication sans la révéler (non persisté)."""

    verdict: Literal["on_track", "partial", "off_track"]
    feedback: str
    hint: str


class SectionNoteRequest(BaseModel):
    note: str = Field("", max_length=2000, description="Note libre de l'apprenant ; chaîne vide pour l'effacer.")

    @field_validator("note")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()


class SectionNoteResponse(BaseModel):
    note: str
    updated_at: str


class VideoNoteRequest(BaseModel):
    note: str = Field("", max_length=2000, description="Note libre de l'apprenant ; chaîne vide pour l'effacer.")

    @field_validator("note")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()


class VideoNoteResponse(BaseModel):
    note: str
    updated_at: str


class MoreSectionsResponse(BaseModel):
    sections: list[ApiPlannedSection]
    next_steps: ApiPlannedSection | None = Field(
        None,
        description=(
            "Section « Pour aller plus loin » actualisée avec de nouvelles pistes (mêmes `title` et `order` "
            "que celle envoyée) : le client doit REMPLACER l'ancienne section par celle-ci. None uniquement si "
            "le plan n'en contenait pas."
        ),
    )


class AddCourseSectionsRequest(BaseModel):
    """Demande d'ajout de contenu à un cours déjà généré.

    `instructions` vide (ou absente) : le sujet vient de `next_steps` si le cours en a, sinon de
    nouveaux sujets proposés par le modèle à partir du cours existant.
    """

    instructions: str | None = Field(None, max_length=_PLAN_INSTRUCTIONS_MAX)

    @field_validator("instructions")
    @classmethod
    def _blank_instructions_to_none(cls, v: str | None) -> str | None:
        v = (v or "").strip()
        return v or None


class SectionRegenerationFailure(BaseModel):
    section_id: str
    title: str
    reason: Literal["invalid_response", "quota", "unavailable"]


class RegenerateIncompleteSectionsResponse(BaseModel):
    """Résultat d'une régénération groupée : succès partiel possible, jamais d'écriture partielle
    d'une section (chaque section régénérée est enregistrée entière, les autres sont inchangées)."""

    regenerated: list[CourseSection] = Field(default_factory=list, description="Sections régénérées et enregistrées.")
    failed: list[SectionRegenerationFailure] = Field(
        default_factory=list, description="Sections tentées sans succès : toujours incomplètes, relançables."
    )
    remaining_section_ids: list[str] = Field(
        default_factory=list,
        description="Sections incomplètes non tentées (arrêt sur quota/indisponibilité, ou plafond par appel atteint).",
    )
    stopped_reason: Literal["quota", "unavailable", "limit"] | None = Field(
        None, description="Pourquoi le traitement s'est arrêté avant la dernière section ; null s'il est allé au bout."
    )


class AddCourseSectionsResponse(BaseModel):
    sections: list[CourseSection]
    next_steps: list[str] = Field(
        default_factory=list,
        description=(
            "Valeur finale de `next_steps` après l'opération (pistes consommées retirées si les "
            "nouvelles sections en sont issues, inchangée sinon) : le client doit REMPLACER l'ancienne "
            "valeur par celle-ci."
        ),
    )


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
    folder: str
    subfolder: str


class CourseHistoryDetail(BaseModel):
    """Détail d'une session de cours (avec gemini_response)."""
    id: UUID
    created_at: str
    question: str
    filenames: list[str]
    mode: str
    folder: str
    subfolder: str
    gemini_response: CourseGenerationResponse


# ─── Dossiers de cours (rangement virtuel : un attribut sur le cours, pas une entité séparée) ──


class SubfolderSummary(BaseModel):
    name: str
    course_count: int


class FolderSummary(BaseModel):
    name: str
    course_count: int
    subfolders: list[SubfolderSummary]


class FolderTarget(BaseModel):
    """Dossier/sous-dossier cible d'un déplacement (cours ou fichier ingéré).

    `subfolder` omis ou vide = sous-dossier par défaut du dossier cible.
    """
    folder: str = Field(..., min_length=1, max_length=200)
    subfolder: str | None = Field(None, max_length=200)

    @field_validator("folder")
    @classmethod
    def strip_folder(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Le nom du dossier ne peut pas être vide")
        return v

    @field_validator("subfolder")
    @classmethod
    def strip_subfolder(cls, v: str | None) -> str | None:
        v = v.strip() if v else None
        return v or None


class MoveCourseFolderRequest(FolderTarget):
    """Body de PUT /courses/history/{session_id}/folder."""


class FolderMoveResult(BaseModel):
    """Nombre d'éléments (cours ou fichiers) reclassés au dossier/sous-dossier par défaut après une
    suppression."""
    moved: int


# ─── Dossiers de fichiers ingérés (même rangement virtuel que les cours) ─────────


class MoveFileFolderRequest(FolderTarget):
    """Body de PUT /pdf/files/{filename}/folder."""


class FileFolderResult(BaseModel):
    filename: str
    folder: str
    subfolder: str

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


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Question de l'apprenant sur le cours (≤ 1000 caractères).")
    section_id: str | None = Field(
        None, max_length=64, description="Section en cours de lecture (facultatif) : prioritaire dans le contexte."
    )
    parent_id: UUID | None = Field(
        None,
        description=(
            "Réponse du tuteur après laquelle s'insère la question (null = racine). Éditer une question = "
            "renvoyer le `parent_id` de la question éditée (nouvelle version). Absent : suite de la "
            "réponse non supprimée la plus récente."
        ),
    )

    @field_validator("message")
    @classmethod
    def _strip_and_bound(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("message ne peut pas être vide")
        max_length = get_settings().chat_message_max_length
        if len(v) > max_length:
            raise ValueError(f"message trop long (max {max_length} caractères)")
        return v


class ChatSource(BaseModel):
    label: str
    reference: str


class ChatMessage(BaseModel):
    id: UUID
    role: Literal["user", "assistant"]
    content: str
    status: Literal["answered", "off_topic"]
    sources: list[ChatSource] = Field(default_factory=list)
    created_at: datetime
    parent_id: UUID | None = Field(
        None,
        description=(
            "Question : réponse précédente (null = racine) ; réponse : sa question. Les questions de même "
            "parent sont des versions l'une de l'autre."
        ),
    )


class ChatQuota(BaseModel):
    limit: int = Field(..., description="Messages autorisés par jour (UTC) pour ce cours.")
    used: int
    remaining: int
    resets_at: datetime = Field(..., description="Prochain minuit UTC (ISO 8601).")


class ChatHistoryResponse(BaseModel):
    messages: list[ChatMessage] = Field(
        ..., description="Messages non supprimés, toutes versions, en ordre chronologique croissant (arbre via parent_id)."
    )
    quota: ChatQuota


class ChatExchangeResponse(BaseModel):
    user_message: ChatMessage
    assistant_message: ChatMessage
    quota: ChatQuota
