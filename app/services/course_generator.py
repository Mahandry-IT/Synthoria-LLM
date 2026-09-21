import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.api.schemas import CourseGenerationResponse, CourseMeta, CourseSource, CourseTable, CourseVideo
from app.core.config import Settings, get_settings
from app.core.exceptions import (
    GeminiInvalidResponseError,
    GeminiUnavailableError,
)
from app.schemas.course_generation import (
    CoverageCompletionSchema,
    CourseGenerationSchema,
    DirectAnswer,
    QuizDifficulty,
    Section,
)
from app.services.gemini_client import GeminiClient
from app.services.vector_store import NumpyVectorStore
from app.services.youtube import candidate_video, verify_videos

logger = logging.getLogger(__name__)

_DEFAULT_INSTRUCTIONS = (
    "Tu es un professeur pédagogue. Réponds en français avec la méthode "
    "Quoi/Pourquoi/Comment, en incluant systématiquement un exemple travaillé complet."
)
_DEFAULT_PLAN_INSTRUCTIONS = (
    "Tu es un architecte pédagogique. Produis en français un plan de cours détaillé, "
    "complet et ordonné par dépendances logiques (structure uniquement, sans contenu rédigé). "
    "Le nombre de sections est un plancher, jamais un plafond."
)
_instructions_cache: dict[str, str] = {}


def _load_instruction(filename: str, fallback: str) -> str:
    """Charge (et met en cache) un fichier d'instructions Gemini, avec repli sur `fallback`."""
    if filename in _instructions_cache:
        return _instructions_cache[filename]
    candidates = [
        Path(__file__).resolve().parents[2] / "instruction" / filename,
        Path(__file__).resolve().parents[1] / "instruction" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            _instructions_cache[filename] = candidate.read_text(encoding="utf-8")
            return _instructions_cache[filename]
    logger.warning("instruction_file_missing_fallback_to_default", extra={"instruction_file": filename})
    _instructions_cache[filename] = fallback
    return fallback


def _get_teacher_instructions() -> str:
    return _load_instruction("course_generation_instructions.md", _DEFAULT_INSTRUCTIONS)


def _get_plan_instructions() -> str:
    return _load_instruction("course_plan_instructions.md", _DEFAULT_PLAN_INSTRUCTIONS)


def _build_context_block(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return "Aucun extrait de fichier pertinent trouvé pour cette question."
    blocks = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        origin = f"{meta.get('filename', 'inconnu')} (page {meta.get('page', '?')})"
        blocks.append(f"[Source fichier: {origin}]\n{chunk['content']}")
    return "\n\n".join(blocks)


def _assert_file_isolation(chunks: list[dict[str, Any]], filename: str | list[str] | None) -> None:
    """Garde-fou contre la fuite de contenu entre fichiers.

    Vérifie qu'aucun chunk retourné n'a un `metadata.filename` hors du
    filtre demandé. Logue un warning en cas de fuite (au lieu de lever,
    pour ne pas transformer un bug de retrieval en panne totale) et
    retire les chunks fautifs de la liste utilisée pour la génération.
    """
    if not filename:
        return
    allowed = {filename} if isinstance(filename, str) else set(filename)
    leaked = [
        c for c in chunks if c.get("metadata", {}).get("filename") not in allowed
    ]
    if leaked:
        logger.warning(
            "course_file_isolation_breach",
            extra={
                "expected_files": list(allowed),
                "leaked_files": sorted({c.get("metadata", {}).get("filename") for c in leaked}),
                "leaked_count": len(leaked),
            },
        )
        chunks[:] = [c for c in chunks if c not in leaked]


def _missing_pages(chunks: list[dict[str, Any]], total_pages: int) -> list[int]:
    """Retourne les numéros de page (1..total_pages) absents des chunks fournis."""
    covered = {
        c.get("metadata", {}).get("page")
        for c in chunks
        if c.get("metadata", {}).get("page") is not None
    }
    return [p for p in range(1, total_pages + 1) if p not in covered]


def compute_quiz_points(questions: list[Any]) -> list[float]:
    """Répartit 20 points sur les questions selon leur difficulté.

    Poids relatifs : facile 1.0, normale 1.5, difficile 2.0. Les points
    bruts sont proportionnels aux poids, puis arrondis au 0.5 le plus proche.
    Le reliquat (écart à 20.0) est ajusté sur la première question difficile
    (ou normale à défaut) pour garantir que la somme == 20.0 exactement.

    Seule la borne inférieure 0.5 est imposée strictement ; la borne
    supérieure 2.0 est une cible — si le nombre de questions est trop faible
    pour atteindre 20 sans la dépasser, la valeur peut excéder 2.0.

    Paramètres:
        questions: liste d'objets ayant un attribut `difficulty` (QuizDifficulty).

    Retour: liste de points (float) dans le même ordre que `questions`.

    Cas limites:
        - Quiz vide → liste vide.
        - N=1 → 20 points sur une seule question.
    """
    if not questions:
        return []

    WEIGHTS: dict[QuizDifficulty, float] = {
        QuizDifficulty.FACILE: 1.0,
        QuizDifficulty.NORMALE: 1.5,
        QuizDifficulty.DIFFICILE: 2.0,
    }

    n = len(questions)
    weights = [WEIGHTS.get(q.difficulty, 1.5) for q in questions]
    total_weight = sum(weights)

    # Points bruts proportionnels aux poids, cible = 20
    raw = [w / total_weight * 20.0 for w in weights]

    # Arrondir au 0.5 le plus proche
    points = [round(r * 2) / 2 for r in raw]

    # Borne inférieure 0.5 (pas de borne supérieure stricte)
    points = [max(0.5, p) for p in points]

    # Ajuster le reliat sur la première question difficile (ou normale)
    current_total = sum(points)
    residual = round((20.0 - current_total) * 2) / 2
    if residual != 0:
        for i, q in enumerate(questions):
            if q.difficulty in (QuizDifficulty.DIFFICILE, QuizDifficulty.NORMALE):
                points[i] = round((points[i] + residual) * 2) / 2
                break

    return points


def _file_sources_from_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "type": "file",
            "label": chunk.get("metadata", {}).get("filename", "document"),
            "reference": f"page {chunk.get('metadata', {}).get('page', '?')}",
        }
        for chunk in chunks
    ]


def _block_to_text(block: Any) -> str:
    """Sérialise un ContentBlock en texte plat pour l'API legacy (quoi/pourquoi/comment).

    Avant ce correctif, seuls les blocks TEXT étaient lus — un block TABLE ou
    FORMULA généré par Gemini juste après une phrase d'intro disparaissait
    silencieusement de la réponse API (aucune erreur, contenu juste absent).
    """
    if block.text:
        return block.text
    if block.table:
        rows = "; ".join(" | ".join(row) for row in block.table.rows)
        return f"{' | '.join(block.table.headers)} — {rows}"
    if block.formula:
        rendered = f"$${block.formula.latex}$$"
        return rendered + (f" ({block.formula.description})" if block.formula.description else "")
    if block.list_items:
        return " ; ".join(block.list_items)
    if block.code:
        # Bloc fencé pour que le frontend l'affiche avec son composant de code.
        return f"```{block.code_language or ''}\n{block.code}\n```"
    return ""


def _text_of(blocks: list[Any]) -> str:
    """Texte aplati des blocs, tableaux exclus (exposés à part sous forme structurée)."""
    return " ".join(t for t in (_block_to_text(b) for b in blocks if not b.table) if t)


def _tables_of(blocks: list[Any]) -> list[CourseTable]:
    return [
        CourseTable(caption=b.table.caption, headers=b.table.headers, rows=b.table.rows)
        for b in blocks
        if b.table and b.table.headers and b.table.rows
    ]


def _all_blocks(section: Section) -> list[Any]:
    """Blocs directs d'une section puis ceux de ses sous-sections, dans l'ordre."""
    return [*section.blocks, *(b for sub in section.subsections for b in sub.blocks)]


def _map_sections_to_course_sections(
    sections: list[Section],
    start_index: int = 0,
) -> list["CourseSection"]:
    """Convertit des sections block-based (Gemini schema) en CourseSection (API).

    Extraite de `_map_schema_to_response` pour être réutilisée par la
    complétion de couverture sans dupliquer la logique de mapping.

    Paramètres:
        sections: sections block-based depuis CourseGenerationSchema.
        start_index: index de départ pour les IDs (défaut 0, utile pour
            la complétion qui ajoute après les sections existantes).

    Retour: liste de CourseSection au format API.
    """
    from app.api.schemas import CourseSection, Step, WorkedExample

    api_sections: list[CourseSection] = []
    _SKIPPED = {"common_pitfalls", "summary", "next_steps"}

    for i, section in enumerate(sections):
        if section.type.value in _SKIPPED:
            continue
        # Extraire les sous-sections (Quoi/Pourquoi/Comment)
        quoi_text = ""
        pourquoi_text = ""
        comment_text = ""
        worked_ex = None

        unmatched: list[str] = []
        for sub in section.subsections:
            sub_text = _text_of(sub.blocks)
            title_lower = sub.title.lower()
            if "pourquoi" in title_lower:
                pourquoi_text = sub_text
            elif "quoi" in title_lower:
                quoi_text = sub_text
            elif "comment" in title_lower:
                comment_text = sub_text
            elif sub_text:
                unmatched.append(sub_text)
            for b in sub.blocks:
                if b.worked_example:
                    worked_ex = WorkedExample(
                        statement=b.worked_example.statement,
                        steps=[Step(id=str(idx + 1), content=s) for idx, s in enumerate(b.worked_example.steps)],
                        result=b.worked_example.result,
                    )

        # Sous-sections au titre non standard : le contenu ne doit pas disparaître
        # (une section entière était sinon supprimée de la réponse). Repli positionnel
        # sur les champs encore vides, le reste est ajouté à `comment`.
        for text in unmatched:
            if not quoi_text:
                quoi_text = text
            elif not pourquoi_text:
                pourquoi_text = text
            elif not comment_text:
                comment_text = text
            else:
                comment_text = f"{comment_text} {text}"

        # Si pas de sous-sections, extraire depuis les blocks directs
        if not section.subsections and section.blocks:
            direct_text = _text_of(section.blocks)
            if section.type.value == "introduction":
                quoi_text = direct_text
            else:
                comment_text = direct_text

        tables = _tables_of(_all_blocks(section))

        if quoi_text or pourquoi_text or comment_text or tables:
            api_sections.append(CourseSection(
                id=str(start_index + i),
                title=section.title,
                quoi=quoi_text,
                pourquoi=pourquoi_text,
                comment=comment_text,
                worked_example=WorkedExample(
                    statement=worked_ex.statement if worked_ex else "",
                    steps=worked_ex.steps if worked_ex else [],
                    result=worked_ex.result if worked_ex else "",
                ),
                tables=tables,
            ))

    return api_sections


_SHINGLE_SIZE = 3


def _shingles(text: str) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.casefold())
    return {tuple(words[i : i + _SHINGLE_SIZE]) for i in range(max(0, len(words) - _SHINGLE_SIZE + 1))}


def _direct_answer_text(direct: DirectAnswer) -> str:
    return " ".join(
        [direct.summary, *direct.key_points, *(t for t in (_block_to_text(b) for b in direct.blocks) if t)]
    )


def answer_intro_overlap(schema: CourseGenerationSchema) -> float:
    """Part (0-1) de la réponse directe recopiée mot à mot (3-grammes) de l'introduction."""
    if schema.direct_answer is None:
        return 0.0
    intro = _intro_text(schema)
    answer = _shingles(_direct_answer_text(schema.direct_answer))
    return len(answer & _shingles(intro)) / len(answer) if answer else 0.0


async def ensure_distinct_direct_answer(
    structured: dict[str, Any],
    *,
    mode: str,
    gemini_client: GeminiClient,
    system_instruction: str,
    context_prompt: str,
    max_overlap: float,
) -> dict[str, Any]:
    """Relance ciblée du seul `direct_answer` s'il reprend le texte de l'introduction.

    Best-effort : toute erreur (validation, Gemini) laisse `structured` inchangé.
    """
    try:
        schema = CourseGenerationSchema.model_validate(_coerce_known_format(dict(structured), mode))
    except Exception:
        return structured
    overlap = answer_intro_overlap(schema)
    if overlap <= max_overlap:
        return structured

    logger.warning("course_answer_duplicates_intro", extra={"overlap": round(overlap, 2)})
    try:
        retried = await gemini_client.format_structured(
            raw_answer=(
                f"{context_prompt}\n\n"
                "La réponse directe précédente recopiait l'introduction. Régénère UNIQUEMENT la réponse directe : "
                "réponds à la question en 2-3 phrases, 3 à 5 points clés et 1 visuel récapitulatif, sans reprendre "
                "le contexte, les prérequis ni la vue d'ensemble de l'introduction.\n\n"
                f"Introduction à NE PAS reprendre :\n{_intro_text(schema)}"
            ),
            system_instruction=system_instruction,
            response_schema=DirectAnswer,
        )
        return {**structured, "direct_answer": DirectAnswer.model_validate(retried).model_dump(mode="json")}
    except Exception:
        logger.warning("course_answer_regeneration_failed", exc_info=True)
        return structured


def _intro_text(schema: CourseGenerationSchema) -> str:
    return " ".join(
        _block_to_text(b) for s in schema.sections if s.type.value == "introduction" for b in _all_blocks(s)
    )


def _map_schema_to_response(schema: CourseGenerationSchema) -> CourseGenerationResponse:
    """Convertit la réponse Gemini (CourseGenerationSchema) en CourseGenerationResponse API.

    Le schema Gemini est block-based (sections > subsections > ContentBlocks).
    On extrait les données pertinentes pour le format API existant.
    """
    _SOURCE_MAP = {"file_chunk": "file", "web": "web"}
    all_sources = [
        CourseSource(
            type=_SOURCE_MAP.get(s.type.value, s.type.value),
            label=s.label,
            reference=s.reference,
        )
        for s in schema.sources
    ]

    # Extraire summary et next_steps depuis les sections (blocs directs ET sous-sections :
    # Gemini range parfois le contenu dans une sous-section, ce qui laissait `summary` vide).
    summary = ""
    next_steps: list[str] = []
    for section in schema.sections:
        if section.type.value == "summary":
            summary = " ".join(t for t in (_block_to_text(b) for b in _all_blocks(section)) if t)
        elif section.type.value == "next_steps":
            for block in _all_blocks(section):
                if block.list_items:
                    next_steps = block.list_items
                    break
            if not next_steps:
                next_steps = [t for t in (_block_to_text(b) for b in _all_blocks(section)) if t]

    # Mapper les sections block-based vers CourseSection (format API)
    from app.api.schemas import CoursePitfall
    api_sections = _map_sections_to_course_sections(schema.sections)
    pitfalls: list[CoursePitfall] = []
    for section in schema.sections:
        if section.type.value == "common_pitfalls":
            for block in _all_blocks(section):
                if block.pitfall:
                    pitfalls.append(CoursePitfall(
                        description=block.pitfall.description,
                        why_it_happens=block.pitfall.why_it_happens,
                        how_to_avoid=block.pitfall.how_to_avoid,
                    ))
                elif block.text:
                    pitfalls.append(CoursePitfall(description=block.text, why_it_happens="", how_to_avoid=""))

    # Réponse directe : générée à part, jamais recopiée d'une section du cours.
    direct = schema.direct_answer
    answer = {
        "summary": direct.summary if direct else summary,
        "blocks": [b.model_dump(mode="json", exclude_none=True) for b in direct.blocks] if direct else [],
        "key_points": direct.key_points if direct else schema.unconfirmed_points,
    }

    # Calculer les points du quiz (déterministe, source de vérité côté serveur)
    quiz_points = compute_quiz_points(schema.quiz) if schema.quiz else []
    quiz_items = []
    for q, pts in zip(schema.quiz, quiz_points):
        quiz_items.append({
            "question": q.question,
            "options": q.choices,
            "correct_option_indices": q.correct_indices,
            "difficulty": q.difficulty.value,
            "points": pts,
            "explanation": q.explanation,
            "time_limit_seconds": 80 if q.requires_calculation else 45,
        })

    return CourseGenerationResponse(
        mode=schema.mode.value,
        format=schema.format.value,
        meta=CourseMeta(
            title=schema.meta.title,
            subject=schema.meta.subject,
            language=schema.meta.language,
            generated_at=schema.meta.generated_at.isoformat() if isinstance(schema.meta.generated_at, datetime) else str(schema.meta.generated_at),
        ),
        sources=all_sources,
        answer=answer,
        sections=api_sections or None,
        common_pitfalls=pitfalls or None,
        quiz=quiz_items or None,
        summary=summary,
        next_steps=next_steps or schema.unconfirmed_points,
        videos=[v for v in (candidate_video(s.url, s.title) for s in schema.video_suggestions) if v],
    )


_YOUTUBE_URL_RE = re.compile(
    r"https?://(?:www\.|m\.)?(?:youtube\.com/(?:watch\?v=|embed/|shorts/)|youtu\.be/)[A-Za-z0-9_-]{11}"
)


async def _search_videos(topic: str, gemini_client: GeminiClient) -> list[CourseVideo]:
    """Recherche web (grounding) de vidéos YouTube sur le sujet ; URL extraites du texte, non vérifiées."""
    raw, _ = await gemini_client.search_grounded(
        prompt=(
            f"Trouve 3 vidéos YouTube pédagogiques, de préférence en français, qui expliquent bien : {topic}. "
            "Donne pour chacune son titre et son URL complète (https://www.youtube.com/watch?v=...). "
            "Ne cite que des vidéos réellement trouvées."
        ),
        system_instruction="Tu es un assistant de recherche de ressources pédagogiques.",
    )
    urls = list(dict.fromkeys(_YOUTUBE_URL_RE.findall(raw)))
    return [v for v in (candidate_video(u) for u in urls) if v]


async def attach_verified_videos(
    response: CourseGenerationResponse, settings: Settings, gemini_client: GeminiClient | None = None,
) -> CourseGenerationResponse:
    """Remplace les vidéos candidates du modèle par celles réellement disponibles sur YouTube.

    Le modèle invente souvent des identifiants : chaque candidat est vérifié (oEmbed). Si aucun
    ne passe, une recherche web dédiée est tentée (un seul appel). Best-effort : ne lève jamais.
    """
    if not settings.course_videos_enabled:
        return response.model_copy(update={"videos": []})

    async def _verified(candidates: list[CourseVideo]) -> list[CourseVideo]:
        return await verify_videos(
            candidates,
            timeout_seconds=settings.course_videos_verify_timeout_seconds,
            max_videos=settings.course_videos_max,
        )

    verified: list[CourseVideo] = []
    try:
        verified = await _verified(response.videos)
        if not verified and gemini_client is not None:
            topic = f"{response.meta.title} ({response.meta.subject})"
            verified = await _verified(await _search_videos(topic, gemini_client))
    except Exception:
        logger.warning("course_videos_lookup_failed", exc_info=True)
    return response.model_copy(update={"videos": verified})


_MODE_TO_FORMAT: dict[str, str] = {
    "file_question": "focused_answer",
    "question_only": "focused_answer",
}


def _coerce_known_format(structured: dict[str, Any], mode: str) -> dict[str, Any]:
    """Corrige les champs `mode` et `format` qui sont entièrement dérivés du
    mode réel passé en paramètre — Gemini a tendance à les halluciner.

    On écrase ces valeurs côté code plutôt que de dépendre du modèle :
    ça élimine la classe d'erreur au lieu de la détecter, et évite de
    perdre un appel Gemini sur une regénération.
    """
    structured["mode"] = mode
    expected = _MODE_TO_FORMAT.get(mode)
    if expected is not None:
        structured["format"] = expected
    return structured


def _is_quiz_difficulty_error(exc: Exception) -> bool:
    """Détecte si l'erreur est liée à la répartition de difficulté du quiz."""
    msg = str(exc).lower()
    return "répartition de difficulté" in msg or "quiz_difficulty" in msg.lower()


def _rebalance_quiz_difficulty(structured: dict[str, Any]) -> dict[str, Any]:
    """Réassigne déterministiquement les labels difficulty du quiz pour respecter
    exactement la distribution attendue (difficile=round(N/2), normale=round(N/4),
    facile=N-reste). Tri stable par index pour un comportement prévisible.

    Cette fonction est le filet de sécurité final : aucune génération ne doit
    échouer sur ce seul critère.
    """
    quiz = structured.get("quiz")
    if not quiz or len(quiz) < 2:
        return structured

    n = len(quiz)
    n_difficile = round(n / 2)
    n_normale = round(n / 4)
    n_facile = n - n_difficile - n_normale

    # Tri stable par index d'origine : on garde l'ordre d'arrivée
    difficulties = ["difficile"] * n_difficile + ["normale"] * n_normale + ["facile"] * n_facile
    for i, q in enumerate(quiz):
        q["difficulty"] = difficulties[i]

    return structured


def _validate_and_map(structured: dict[str, Any], mode: str) -> CourseGenerationResponse:
    """Corrige le format déterministe, valide contre CourseGenerationSchema,
    puis mappe vers CourseGenerationResponse — sous un seul try/except.
    """
    try:
        structured = _coerce_known_format(structured, mode)
        gemini_result = CourseGenerationSchema.model_validate(structured)
        return _map_schema_to_response(gemini_result)
    except Exception as exc:
        if _is_quiz_difficulty_error(exc):
            # Fallback déterministe : rééquilibre sans appel LLM supplémentaire
            structured = _rebalance_quiz_difficulty(structured)
            structured = _coerce_known_format(structured, mode)
            try:
                gemini_result = CourseGenerationSchema.model_validate(structured)
                return _map_schema_to_response(gemini_result)
            except Exception:
                pass  # passer au raise final ci-dessous
        raise GeminiInvalidResponseError(f"JSON structuré invalide: {exc}") from exc


async def _validate_and_map_with_retry(
    structured: dict[str, Any],
    mode: str,
    gemini_client: GeminiClient,
    system_instruction: str,
    raw_answer_for_retry: str,
) -> CourseGenerationResponse:
    """Wrapper async de _validate_and_map avec retry ciblé Gemini sur erreur
    de répartition de difficulté du quiz.

    Stratégie :
      1. Essaie _validate_and_map (inclut déjà le fallback déterministe).
      2. Si échec GeminiInvalidResponseError sur difficulté → 1 retry Gemini
         avec message d'erreur exact, en demandant de corriger uniquement
         le champ difficulty.
      3. Si le retry échoue aussi → fallback déterministe garanti.
    """
    structured = await ensure_distinct_direct_answer(
        structured, mode=mode, gemini_client=gemini_client, system_instruction=system_instruction,
        context_prompt=raw_answer_for_retry, max_overlap=get_settings().course_answer_intro_similarity_max,
    )
    try:
        return _validate_and_map(structured, mode)
    except GeminiInvalidResponseError as exc:
        if not _is_quiz_difficulty_error(exc):
            raise
        # Retry ciblé Gemini (1 tentative)
        retry_prompt = (
            f"{raw_answer_for_retry}\n\n"
            f"ERREUR DE VALIDATION : {exc}\n"
            f"Corrige UNIQUEMENT le champ 'difficulty' des questions concernées "
            f"pour respecter la distribution : difficile=round(N/2), "
            f"normale=round(N/4), facile=N-difficile-normale. "
            f"Ne change ni les questions ni les réponses."
        )
        try:
            retried = await gemini_client.format_structured(
                raw_answer=retry_prompt,
                system_instruction=system_instruction,
                response_schema=CourseGenerationSchema,
            )
            return _validate_and_map(retried, mode)
        except Exception:
            pass  # Fallback déterministe final (déjà dans _validate_and_map)
        # Dernier filet : rebalance sur le structured original
        structured = _rebalance_quiz_difficulty(structured)
        return _validate_and_map(structured, mode)


def _build_coverage_completion_prompt(
    missing_chunks: list[dict[str, Any]],
    existing_titles: list[str],
) -> str:
    """Construit le prompt pour l'appel Gemini de complétion de couverture.

    Paramètres:
        missing_chunks: chunks non couverts (texte + métadonnées fichier/page).
        existing_titles: titres des sections déjà générées, pour que Gemini
            décide fusion vs nouvelle section.

    Retour: prompt texte prêt à être passé à format_structured.
    """
    chunks_text = "\n\n".join(
        f"[{c.get('metadata', {}).get('filename')} p.{c.get('metadata', {}).get('page')}] {c['content']}"
        for c in missing_chunks
    )
    titles_str = "\n".join(f"- {t}" for t in existing_titles) if existing_titles else "(aucune section existante)"
    return (
        "Tu dois intégrer le contenu manquant ci-dessous dans le cours. "
        "Pour chaque extrait : "
        "1. Si le sujet correspond à une section existante (liste ci-dessous), "
        "crée une section avec le MÊME titre pour permettre la fusion.\n"
        "2. Sinon, crée une NOUVELLE section thématique avec un titre réel et précis.\n\n"
        "INTERDICTION ABSOLUE : tout titre générique du type 'Contenu complémentaire', "
        "'Pages non couvertes', 'Supplément', 'Section additionnelle', etc. "
        "Chaque section DOIT porter un titre thématique réel et descriptif.\n\n"
        f"--- Sections existantes ---\n{titles_str}\n\n"
        f"--- Contenu manquant à intégrer ---\n{chunks_text}"
    )


async def _complete_missing_coverage(
    response: CourseGenerationResponse,
    missing_chunks: list[dict[str, Any]],
    gemini_client: GeminiClient,
    settings: Settings,
    system_instruction: str,
) -> CourseGenerationResponse:
    """Appel Gemini conditionnel pour intégrer le contenu manquant dans des
    sections réelles (fusion ou nouvelles sections thématiques).

    Comportement :
    - Si le flag course_coverage_completion_enabled est désactivé → retour inchangé.
    - Si le volume cumulé est inférieur au seuil → retour inchangé.
    - En cas d'exception (Gemini indisponible, quota, JSON invalide) →
      logger.warning + retour inchangé (best-effort non-bloquant).

    Paramètres:
        response: réponse API courante (à enrichir).
        missing_chunks: chunks non couverts.
        gemini_client: client Gemini.
        settings: configuration applicative.
        system_instruction: instructions système pour Gemini.

    Retour: CourseGenerationResponse potentiellement enrichi.
    """
    if not settings.course_coverage_completion_enabled:
        return response

    total_missing_chars = sum(len(c.get("content", "")) for c in missing_chunks)
    if total_missing_chars < settings.course_coverage_min_missing_chars:
        logger.info(
            "course_coverage_below_threshold",
            extra={"missing_chars": total_missing_chars, "threshold": settings.course_coverage_min_missing_chars},
        )
        return response

    existing_titles = [s.title for s in (response.sections or [])]
    prompt = _build_coverage_completion_prompt(missing_chunks, existing_titles)

    try:
        structured = await gemini_client.format_structured(
            raw_answer=prompt,
            system_instruction=system_instruction,
            response_schema=CoverageCompletionSchema,
        )
        completion = CoverageCompletionSchema.model_validate(structured)
    except Exception as exc:
        logger.warning("course_coverage_completion_failed", extra={"error": str(exc)})
        return response

    new_sections = _map_sections_to_course_sections(
        completion.sections,
        start_index=len(response.sections or []),
    )

    # Fusion par titre : si le titre normalisé correspond déjà → concatener comment
    merged: list = list(response.sections or [])
    existing_normalized = {s.title.strip().casefold(): s for s in merged}
    new_sources: list[CourseSource] = []

    for ns in new_sections:
        key = ns.title.strip().casefold()
        if key in existing_normalized:
            existing_section = existing_normalized[key]
            existing_section.comment = (
                (existing_section.comment + "\n\n" + ns.comment).strip()
            )
            existing_section.tables = [*existing_section.tables, *ns.tables]
            if not existing_section.quoi and ns.quoi:
                existing_section.quoi = ns.quoi
            if not existing_section.pourquoi and ns.pourquoi:
                existing_section.pourquoi = ns.pourquoi
        else:
            merged.append(ns)
            existing_normalized[key] = ns

    # Ajouter les sources correspondantes aux chunks absorbés
    for c in missing_chunks:
        new_sources.append(
            CourseSource(
                type="file",
                label=c.get("metadata", {}).get("filename", "document"),
                reference=f"page {c.get('metadata', {}).get('page', '?')}",
            )
        )

    return response.model_copy(update={
        "sections": merged,
        "sources": response.sources + new_sources,
    })


async def _apply_coverage_check(
    response: CourseGenerationResponse,
    vector_store: NumpyVectorStore,
    filename: str | list[str] | None,
    gemini_client: GeminiClient | None = None,
    settings: Settings | None = None,
    system_instruction: str = "",
) -> CourseGenerationResponse:
    """Contrôle de couverture post-génération (une seule passe, pas de boucle).

    Compare les pages citées dans `sources` (type "file") aux pages totales
    du/des fichier(s) (`vector_store.count_pages`). Les pages manquantes sont
    récupérées via `get_all_chunks` puis intégrées dans des sections réelles
    via un appel Gemini de complétion (best-effort, non-bloquant).

    Si gemini_client/settings ne sont pas fournis, fallback sur le comportement
    legacy (dump brut) pour compatibilité.

    Lève: rien. En cas d'échec, la réponse courante est retournée inchangée.
    """
    if not filename:
        return response

    filenames = [filename] if isinstance(filename, str) else list(filename)
    cited_pages_by_file: dict[str, set[int]] = {f: set() for f in filenames}
    for src in response.sources:
        if src.type != "file" or src.label not in cited_pages_by_file:
            continue
        match = re.search(r"\d+", src.reference)
        if match:
            cited_pages_by_file[src.label].add(int(match.group()))

    missing_chunks: list[dict[str, Any]] = []
    missing_pages_log: dict[str, list[int]] = {}
    for fname in filenames:
        total_pages = vector_store.count_pages(fname)
        if total_pages == 0:
            continue
        missing = [p for p in range(1, total_pages + 1) if p not in cited_pages_by_file[fname]]
        if not missing:
            continue
        missing_pages_log[fname] = missing
        missing_chunks.extend(
            c for c in vector_store.get_all_chunks(fname) if c.get("metadata", {}).get("page") in missing
        )

    if not missing_chunks:
        logger.info("course_coverage_check", extra={"missing_pages": {}})
        return response

    logger.info("course_coverage_check", extra={"missing_pages": missing_pages_log})

    # Nouveau comportement : complétion réelle via Gemini (best-effort)
    if gemini_client is not None and settings is not None:
        return await _complete_missing_coverage(
            response, missing_chunks, gemini_client, settings, system_instruction
        )

    # Fallback legacy si gemini_client/settings non fournis (compatibilité)
    from app.api.schemas import CourseSection, WorkedExample

    supplement_text = "\n\n".join(
        f"[{c.get('metadata', {}).get('filename')} p.{c.get('metadata', {}).get('page')}] {c['content']}"
        for c in missing_chunks
    )
    supplement_section = CourseSection(
        id=f"coverage-supplement-{len(response.sections or [])}",
        title="Contenu complémentaire (pages non couvertes initialement)",
        quoi="",
        pourquoi="",
        comment=supplement_text,
        worked_example=WorkedExample(statement="", steps=[], result=""),
    )
    supplement_sources = [
        CourseSource(
            type="file",
            label=c.get("metadata", {}).get("filename", "document"),
            reference=f"page {c.get('metadata', {}).get('page', '?')}",
        )
        for c in missing_chunks
    ]

    return response.model_copy(update={
        "sections": (response.sections or []) + [supplement_section],
        "sources": response.sources + supplement_sources,
    })


async def _retrieve_chunks(
    question: str,
    vector_store: NumpyVectorStore,
    gemini_client: GeminiClient,
    settings: Settings,
    mode: str,
    top_k: int | None,
    filename: str | list[str] | None,
    full_document: bool,
) -> tuple[str, list[dict[str, Any]]]:
    """Reformule la question puis récupère les chunks du vector store.

    Partagée par la génération directe et la planification.

    Retour: (requête reformulée, chunks filtrés par isolation fichier).
    Vide en mode `question_only`.
    """
    resolved_top_k = top_k or settings.course_top_k_default

    # Reformulation de la query pour améliorer le matching sémantique
    search_query = await gemini_client.reformulate_query(question, filename)

    chunks: list[dict[str, Any]] = []
    if mode != "question_only":
        if full_document and filename:
            chunks = vector_store.get_all_chunks(filename)
        # Si plusieurs fichiers : recherche séparée par fichier pour assurer
        # une représentation équilibrée de chaque document.
        elif isinstance(filename, list) and len(filename) > 1:
            per_file_k = max(resolved_top_k // len(filename), 5)
            all_chunks: list[dict[str, Any]] = []
            for fname in filename:
                # Le filtre filename_filter restreint les candidats AVANT le
                # classement top_k : chaque fichier obtient ses propres chunks
                # les plus pertinents, au lieu de puiser dans un même top_k
                # global (ce qui écrasait les fichiers non dominants).
                file_chunks = await vector_store.search(
                    search_query, top_k=per_file_k, filename_filter=fname
                )
                all_chunks.extend(file_chunks)
            # Dé-duplication (un chunk peut matcher dans plusieurs recherches)
            seen: set[str] = set()
            chunks = []
            for c in all_chunks:
                # Les chunks n'ont pas de champ 'id' dans les résultats de search
                cid = c.get("content", "")[:200]
                if cid not in seen:
                    seen.add(cid)
                    chunks.append(c)
            # Limiter au top_k global
            chunks = chunks[:resolved_top_k]
        else:
            chunks = await vector_store.search(
                search_query, top_k=resolved_top_k, filename_filter=filename
            )

    _assert_file_isolation(chunks, filename)
    return search_query, chunks


async def generate_course_from_question(
    question: str,
    vector_store: NumpyVectorStore,
    gemini_client: GeminiClient,
    settings: Settings,
    mode: str = "file_question",
    top_k: int | None = None,
    filename: str | list[str] | None = None,
    full_document: bool = False,
) -> CourseGenerationResponse:
    """
    Orchestration RAG + génération de cours structuré.

    Paramètres:
        question: question de l'utilisateur.
        vector_store: store vectoriel local (retrieval top-k).
        gemini_client: client Gemini.
        settings: configuration applicative.
        mode: "file_question" (Mode 2, retrieval RAG) ou "question_only" (Mode 3, recherche web).
        top_k: nombre de chunks à récupérer (défaut settings.course_top_k_default).
        filename: filtre optionnel — un nom (str), une liste de noms, ou None.
        full_document: si True, ignore le top-k et récupère TOUS les chunks du/des
            fichier(s) (`vector_store.get_all_chunks`) pour une couverture exhaustive,
            au prix d'un contexte plus volumineux envoyé à Gemini.

    Retour: CourseGenerationResponse validé.

    Fonctionnement:
        1. Retrieval des chunks pertinents dans le vector store local (sauf mode question_only).
           - Si full_document=True : tous les chunks du/des fichier(s), triés par page.
           - Si filename est une liste : recherche séparée par fichier + fusion équilibrée.
           - Si filename est un str ou None : recherche unique.
           Isolation stricte : tout chunk hors du filtre `filename` demandé est
           écarté et logué (`course_file_isolation_breach`).
        2. Si gemini_use_search_grounding=True : 2 appels (search_grounded + format_structured).
           Sinon : 1 seul appel (format_structured direct avec contexte RAG).
        3. Contrôle de couverture post-génération (`_apply_coverage_check`) : les
           pages non citées dans `sources` sont intégrées dans des sections réelles
           via un appel Gemini additionnel (best-effort, non-bloquant). Seuil
           configurable (`course_coverage_min_missing_chars`) et coupe-circuit
           (`course_coverage_completion_enabled`).

    Lève: GeminiUnavailableError, GeminiQuotaExceededError, GeminiInvalidResponseError.
    """
    search_query, chunks = await _retrieve_chunks(
        question, vector_store, gemini_client, settings, mode, top_k, filename, full_document
    )

    context_block = _build_context_block(chunks)
    file_sources = _file_sources_from_chunks(chunks)
    system_instruction = _get_teacher_instructions()

    is_question_only = mode == "question_only"

    # --- Mode 1 appel (pas de recherche web, quota minima) ---
    if not settings.gemini_use_search_grounding:
        if is_question_only:
            prompt = (
                f'mode="{mode}"\n'
                f"Question de l'utilisateur : {question}\n\n"
                f"Sources fichier disponibles : []\n"
                f"Sources web disponibles : []\n\n"
                f"Génère directement le JSON structuré selon le schéma fourni."
            )
        else:
            prompt = (
                f'mode="{mode}"\n'
                f"Question de l'utilisateur : {question}\n\n"
                f"Contexte extrait des documents fournis :\n{context_block}\n\n"
                f"Sources fichier disponibles : {file_sources}\n"
                f"Sources web disponibles : []\n\n"
                f"Génère directement le JSON structuré selon le schéma fourni."
            )
        structured = await gemini_client.format_structured(
            raw_answer=prompt,
            system_instruction=system_instruction,
            response_schema=CourseGenerationSchema,
        )
        validated = await _validate_and_map_with_retry(
            structured, mode, gemini_client, system_instruction, prompt,
        )
        completed = await _apply_coverage_check(
            validated, vector_store, filename,
            gemini_client=gemini_client, settings=settings,
            system_instruction=system_instruction,
        )
        return await attach_verified_videos(completed, settings, gemini_client)

    # --- Mode 2 appels (search grounding + reformatage) ---
    if is_question_only:
        prompt = f"Question de l'utilisateur : {question}"
    else:
        prompt = (
            f"Question de l'utilisateur : {question}\n\n"
            f"Contexte extrait des documents fournis (à compléter par une recherche web si nécessaire) :\n"
            f"{context_block}"
        )

    raw_answer, web_sources = await gemini_client.search_grounded(
        prompt=prompt,
        system_instruction=system_instruction,
    )

    formatting_prompt = (
        f'mode="{mode}"\n'
        f"Sources fichier disponibles : {file_sources}\n"
        f"Sources web disponibles : {web_sources}\n\n"
        f"Réponse brute à structurer en JSON selon le schéma fourni :\n{raw_answer}"
    )

    structured = await gemini_client.format_structured(
        raw_answer=formatting_prompt,
        system_instruction=system_instruction,
        response_schema=CourseGenerationSchema,
    )
    validated = await _validate_and_map_with_retry(
        structured, mode, gemini_client, system_instruction, formatting_prompt,
    )
    completed = await _apply_coverage_check(
        validated, vector_store, filename,
        gemini_client=gemini_client, settings=settings,
        system_instruction=system_instruction,
    )
    return await attach_verified_videos(completed, settings, gemini_client)
