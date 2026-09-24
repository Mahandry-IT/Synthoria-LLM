"""Génération de cours en deux temps : plan validable → cours complet.

1. `generate_course_plan` : structure du cours (sans contenu rédigé) + contexte
   de récupération figé, persisté par l'appelant.
2. `generate_course_from_validated_plan` : cours complet à partir du plan validé
   (éventuellement édité), généré **par lots** de sections DEVELOPMENT pour
   éviter la troncature JSON / la dégradation qualité sur les cours longs.

Réutilise les helpers de `course_generator` (retrieval, mapping, quiz).
"""

import logging
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.api.schemas import (
    _PLAN_OBJECTIVE_MAX,
    _PLAN_SUBTOPIC_MAX,
    _PLAN_SUBTOPICS_MAX_ITEMS,
    _PLAN_TITLE_MAX,
    ApiPlannedSection,
    CourseGenerationResponse,
)
from app.core.config import Settings, get_settings
from app.core.exceptions import GeminiInvalidResponseError, GeminiServiceError
from app.core.markdown import markdown_to_plain
from app.db.models import CoursePlan
from app.schemas.course_generation import (
    BlockType,
    ContentBlock,
    CourseGenerationSchema,
    CoursePlanSchema,
    MoreSectionsSchema,
    NextStepsSchema,
    PlannedSection,
    Section,
    SectionsBatchSchema,
    SectionType,
    Source,
    SourceType,
    Subsection,
)
from app.services.course_generator import (
    INCOMPLETE_SECTION_NOTICE,
    _build_context_block,
    _coerce_known_format,
    ensure_distinct_direct_answer,
    _file_sources_from_chunks,
    _get_plan_instructions,
    _get_teacher_instructions,
    _is_quiz_difficulty_error,
    _rebalance_quiz_difficulty,
    _retrieve_chunks,
    _validate_and_map,
    is_incomplete_section,
)
from app.services.course_videos import attach_verified_videos
from app.services.gemini_client import GeminiClient
from app.services.media.visual_resolver import resolve_visuals
from app.services.vector_store import NumpyVectorStore
from app.services.visual_validation import visual_issues

logger = logging.getLogger(__name__)

_NO_CONTEXT = "Aucun contexte source fourni : s'appuyer sur les connaissances générales et signaler toute incertitude."
_INCOMPLETE_NOTICE = INCOMPLETE_SECTION_NOTICE  # ré-export local : nombreux appels internes à ce module
_WRAP_UP_FAILED_NOTE = (
    "Introduction, pièges courants, résumé et quiz non générés (erreur temporaire) : "
    "relancez la génération du cours."
)


# ─── Contexte figé ───────────────────────────────────────────


def _render_context(retrieval_context: dict[str, Any]) -> str:
    """Reconstitue le bloc de contexte à partir du contexte figé lors de la planification."""
    parts: list[str] = []
    if retrieval_context.get("chunks"):
        parts.append(_build_context_block(retrieval_context["chunks"]))
    web_research = retrieval_context.get("web_research")
    if web_research:
        parts.append(f"Synthèse de recherche web :\n{web_research['raw_answer']}")
    return "\n\n".join(parts) or _NO_CONTEXT


def _sources_from_context(retrieval_context: dict[str, Any]) -> list[Source]:
    """Sources du cours, dérivées déterministiquement du contexte figé (pas du modèle)."""
    seen: set[tuple[str, str]] = set()
    sources: list[Source] = []
    candidates = [
        (SourceType.FILE_CHUNK, s) for s in retrieval_context.get("file_sources", [])
    ] + [
        (SourceType.WEB, s)
        for s in (retrieval_context.get("web_research") or {}).get("web_sources", [])
    ]
    for source_type, raw in candidates:
        label = raw.get("label") or raw.get("reference") or "source"
        reference = raw.get("reference", "")
        if (label, reference) in seen:
            continue
        seen.add((label, reference))
        sources.append(Source(type=source_type, label=label, reference=reference))
    return sources


# ─── Étape 1 : plan ──────────────────────────────────────────


def _normalize_plan(plan: CoursePlanSchema) -> CoursePlanSchema:
    """Trie par `order` (stable) puis renumérote 1..n pour un ordre strictement consécutif."""
    ordered = sorted(plan.planned_sections, key=lambda s: s.order)
    renumbered = [
        s.model_copy(update={"title": s.title.strip(), "order": i})
        for i, s in enumerate(ordered, start=1)
    ]
    return plan.model_copy(update={"planned_sections": renumbered})


async def generate_course_plan(
    question: str,
    vector_store: NumpyVectorStore,
    gemini_client: GeminiClient,
    settings: Settings,
    mode: str = "file_question",
    top_k: int | None = None,
    filename: str | list[str] | None = None,
    full_document: bool = False,
) -> tuple[CoursePlanSchema, dict[str, Any]]:
    """Génère le plan (structure) d'un cours, sans contenu Quoi/Pourquoi/Comment.

    Paramètres:
        question: question de l'utilisateur.
        vector_store: store vectoriel local.
        gemini_client: client Gemini.
        settings: configuration applicative.
        mode: "file_question" (RAG) ou "question_only" (recherche web si grounding activé).
        top_k, filename, full_document: paramètres de retrieval (cf. `generate_course_from_question`).

    Retour: (plan normalisé, contexte de récupération figé à persister). Le contexte
    contient `chunks`, `file_sources`, `search_query` et `web_research` (None hors
    mode question_only avec grounding).

    Fonctionnement: retrieval identique à la génération directe ; en mode
    question_only avec grounding, 1 appel de recherche web dont la synthèse est
    figée pour servir de contexte aux lots (sinon les lots n'auraient aucune
    source) ; puis 1 appel `format_structured` avec `CoursePlanSchema`.

    Lève: GeminiUnavailableError, GeminiQuotaExceededError, GeminiInvalidResponseError.
    """
    search_query, chunks = await _retrieve_chunks(
        question, vector_store, gemini_client, settings, mode, top_k, filename, full_document
    )

    web_research: dict[str, Any] | None = None
    if mode == "question_only" and settings.gemini_use_search_grounding:
        raw_answer, web_sources = await gemini_client.search_grounded(
            prompt=(
                "Recherche les notions essentielles, mécanismes, formules et cas d'usage du sujet "
                f"suivant, pour permettre de bâtir un plan de cours complet.\nSujet : {question}"
            ),
            system_instruction=_get_plan_instructions(),
        )
        web_research = {"raw_answer": raw_answer, "web_sources": web_sources}

    retrieval_context: dict[str, Any] = {
        # Seuls content/metadata sont conservés : `distance` est un artefact de ranking non sérialisable de façon fiable.
        "chunks": [{"content": c["content"], "metadata": c.get("metadata", {})} for c in chunks],
        "file_sources": _file_sources_from_chunks(chunks),
        "search_query": search_query,
        "web_research": web_research,
    }

    prompt = (
        f'mode="{mode}"\n'
        f"Question de l'utilisateur : {question}\n\n"
        f"Contexte :\n{_render_context(retrieval_context)}\n\n"
        "Conçois le plan du cours et retourne-le en JSON selon le schéma fourni (structure uniquement)."
    )
    structured = await gemini_client.format_structured(
        raw_answer=prompt,
        system_instruction=_get_plan_instructions(),
        response_schema=CoursePlanSchema,
    )
    try:
        plan = CoursePlanSchema.model_validate(structured)
    except ValidationError as exc:
        raise GeminiInvalidResponseError(f"Plan structuré invalide: {exc}") from exc

    return _normalize_plan(plan), retrieval_context


# ─── Étape 2 : cours complet à partir du plan ────────────────


def _format_section(section: ApiPlannedSection, *, detailed: bool) -> str:
    line = f"{section.order}. [{section.type}] {section.title}"
    if not detailed:
        return line
    if section.mastery == "known":
        line += (
            "\n   ⚑ Déjà maîtrisée par l'apprenant (pré-test réussi) : version CONDENSÉE — un rappel court, "
            "sans défi ni exemple à trous ; garde tout de même un bloc visuel et une consigne de reformulation."
        )
    if section.objective:
        line += f"\n   Objectif : {section.objective}"
    if section.subtopics:
        line += "\n   Sous-thèmes : " + " ; ".join(section.subtopics)
    return line


def _format_sections(sections: list[ApiPlannedSection], *, detailed: bool) -> str:
    return "\n".join(_format_section(s, detailed=detailed) for s in sections)


def _chunked(items: list[ApiPlannedSection], size: int) -> list[list[ApiPlannedSection]]:
    step = max(1, size)
    return [items[i : i + step] for i in range(0, len(items), step)]


def _has_content(section: Section) -> bool:
    return bool(section.blocks) or any(sub.blocks for sub in section.subsections)


def _incomplete_section(planned: ApiPlannedSection) -> Section:
    """Section de remplacement, visible comme incomplète, quand la génération de son lot a échoué."""

    def text(value: str) -> list[ContentBlock]:
        return [ContentBlock(type=BlockType.TEXT, text=value)]

    # Objectif et sous-thèmes viennent de l'éditeur riche (Markdown) : affichés ici en texte brut.
    subsections = [Subsection(title="Quoi", blocks=text(markdown_to_plain(planned.objective) or planned.title))]
    if planned.subtopics:
        points = " ; ".join(markdown_to_plain(topic) for topic in planned.subtopics)
        subsections.append(Subsection(title="Pourquoi", blocks=text("Points prévus : " + points)))
    subsections.append(Subsection(title="Comment", blocks=text(_INCOMPLETE_NOTICE)))
    return Section(type=SectionType.DEVELOPMENT, title=planned.title, subsections=subsections)


def _align_batch_sections(returned: list[Section], planned: list[ApiPlannedSection]) -> list[Section]:
    """Force la correspondance exacte avec le plan : même nombre, mêmes titres, même ordre.

    Chaque section planifiée est associée à la section retournée de même titre,
    à défaut de même position. Une section absente ou vide devient une section
    « incomplète » plutôt que de disparaître silencieusement.
    """
    by_title = {s.title.strip().casefold(): s for s in returned}
    used: set[int] = set()
    aligned: list[Section] = []

    for index, plan_section in enumerate(planned):
        candidate = by_title.get(plan_section.title.strip().casefold())
        if candidate is None and index < len(returned):
            candidate = returned[index]
        if candidate is None or id(candidate) in used or not _has_content(candidate):
            aligned.append(_incomplete_section(plan_section))
            continue
        used.add(id(candidate))
        aligned.append(candidate.model_copy(update={"type": SectionType.DEVELOPMENT, "title": plan_section.title}))

    if len(returned) != len(planned):
        logger.warning(
            "course_plan_batch_size_mismatch",
            extra={"expected": len(planned), "returned": len(returned)},
        )
    return aligned


def _align_wrap_up_titles(returned: list[Section], planned: list[ApiPlannedSection]) -> list[Section]:
    """Impose les titres du plan aux sections hors développement.

    Chaque section planifiée (introduction, pièges, résumé, suite) reprend le titre
    validé par l'utilisateur ; les sections en trop ou de type absent du plan sont
    écartées (le plan fait foi), et un manque est journalisé.
    """
    remaining = list(returned)
    aligned: list[Section] = []
    for plan_section in planned:
        match = next((s for s in remaining if s.type.value == plan_section.type), None)
        if match is None:
            logger.warning("course_plan_wrap_up_section_missing", extra={"type": plan_section.type})
            continue
        remaining.remove(match)
        aligned.append(match.model_copy(update={"title": plan_section.title}))
    return aligned


async def _generate_batch(
    batch: list[ApiPlannedSection],
    *,
    question: str,
    mode: str,
    context_block: str,
    outline: str,
    gemini_client: GeminiClient,
) -> list[Section]:
    prompt = (
        f'mode="{mode}"\n'
        f"Question de l'utilisateur : {question}\n\n"
        f"Contexte source (figé lors de la planification) :\n{context_block}\n\n"
        f"--- Plan complet validé (pour éviter les doublons et garder la cohérence ; ne développe PAS les sections hors lot) ---\n{outline}\n\n"
        f"--- Sections à générer dans CE lot, et seulement celles-ci ---\n{_format_sections(batch, detailed=True)}\n\n"
        f"Génère exactement {len(batch)} section(s) DEVELOPMENT, dans cet ordre et avec ces titres, "
        "en JSON selon le schéma fourni. Chaque sous-thème listé doit être EXPLIQUÉ (pas seulement cité) ; "
        "renseigne `covered_subtopics` avec les sous-thèmes réellement développés, recopiés à l'identique."
    )
    structured = await gemini_client.format_structured(
        raw_answer=prompt,
        system_instruction=_get_teacher_instructions(),
        response_schema=SectionsBatchSchema,
    )
    aligned = _align_batch_sections(SectionsBatchSchema.model_validate(structured).sections, batch)
    repaired = await _repair_incomplete_sections(
        aligned, batch, question=question, mode=mode, context_block=context_block,
        outline=outline, gemini_client=gemini_client,
    )
    return _finalize_incomplete(repaired, batch)


_DEVELOPMENT_SUBSECTIONS = ("Pourquoi", "Quoi", "Comment")


def _structural_gaps(section: Section) -> list[str]:
    """Sous-sections Pourquoi/Quoi/Comment sans aucun bloc — jamais détecté avant : `_has_content`
    n'exige qu'un bloc *quelque part* dans la section, pas dans chacune des trois sous-sections
    exigées par les instructions."""
    with_blocks = {sub.title for sub in section.subsections if sub.blocks}
    return [name for name in _DEVELOPMENT_SUBSECTIONS if name not in with_blocks]


def _section_text(section: Section) -> str:
    """Concatène tout le texte exploitable d'une section (blocs texte, tableaux, formules, légendes,
    diagrammes, graphiques) — utilisé pour la vérification déterministe de couverture (repli quand
    `covered_subtopics` n'a pas été rempli par le modèle)."""
    parts: list[str] = []
    for sub in section.subsections:
        for b in sub.blocks:
            if b.text:
                parts.append(b.text)
            if b.image_caption:
                parts.append(b.image_caption)
            if b.table:
                parts.append(b.table.caption)
                parts.extend(b.table.headers)
                for row in b.table.rows:
                    parts.extend(row)
            if b.formula:
                parts.append(b.formula.latex)
                if b.formula.description:
                    parts.append(b.formula.description)
            if b.diagram:
                parts.append(b.diagram.caption)
            if b.chart:
                parts.append(b.chart.caption)
                parts.extend(b.chart.labels)
    return " ".join(parts)


def _norm(text: str) -> str:
    """Forme comparable : sans Markdown (saisie de l'éditeur riche), accents, casse ni ponctuation."""
    stripped = unicodedata.normalize("NFKD", markdown_to_plain(text).casefold())
    stripped = "".join(c for c in stripped if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^\w\s]", " ", stripped).split())


def _missing_subtopics(section: Section, planned: ApiPlannedSection) -> list[str]:
    """Sous-thèmes du plan que la section ne traite pas.

    Priorité à la déclaration du modèle (`covered_subtopics`, plus fiable qu'un mot-clé). Si elle
    n'a pas été remplie mais que la section a du contenu, repli sur une vérification déterministe
    du texte réel — auparavant l'absence de déclaration désactivait purement le contrôle, laissant
    passer des sous-thèmes jamais traités. Une section sans aucun contenu (échec de génération,
    repli déjà marqué incomplet) ne déclenche jamais ce contrôle coûteux.
    """
    if not _has_content(section):
        return []
    declared = [_norm(s) for s in section.covered_subtopics if s.strip()]
    if declared:
        return [
            topic
            for topic in planned.subtopics
            if not any(_norm(topic) in d or d in _norm(topic) for d in declared)
        ]
    content = _norm(_section_text(section))
    missing = []
    for topic in planned.subtopics:
        words = [w for w in _norm(topic).split() if len(w) >= 4]
        if words and sum(1 for w in words if w in content) / len(words) < 0.5:
            missing.append(topic)
    return missing


def _completeness_issues(section: Section, planned: ApiPlannedSection) -> list[str]:
    """Tous les problèmes de complétude d'une section par rapport au plan : sous-section
    Pourquoi/Quoi/Comment vide, sous-thème du plan non traité, absence de bloc visuel. Une section
    de repli déjà marquée incomplète (voir `is_incomplete_section`) n'est jamais re-signalée ici :
    son remplacement relève de la régénération explicite par l'apprenant, pas de cette boucle."""
    if is_incomplete_section(section):
        return []
    issues = [f"sous-section « {name} » vide, à écrire" for name in _structural_gaps(section)]
    missing_topics = _missing_subtopics(section, planned)
    if missing_topics:
        issues.append("sous-thèmes non traités : " + " ; ".join(missing_topics))
    issues.extend(visual_issues(section))
    return issues


_MAX_REPAIR_ATTEMPTS = 2


async def _repair_incomplete_sections(
    sections: list[Section],
    batch: list[ApiPlannedSection],
    *,
    question: str,
    mode: str,
    context_block: str,
    outline: str,
    gemini_client: GeminiClient,
) -> list[Section]:
    """Répare les sections du lot ayant des `_completeness_issues`, dans la limite de
    `_MAX_REPAIR_ATTEMPTS` appels Gemini (jamais une boucle illimitée — coût de quota borné).

    S'arrête dès que le lot est complet. Un remplaçant n'est retenu que s'il a strictement moins
    de problèmes que l'original ; en cas d'échec Gemini, les sections en cours sont conservées
    telles quelles. Les sections encore incomplètes après la dernière tentative restent gérées par
    `_finalize_incomplete` (jamais livrées silencieusement avec des trous).
    """
    current = list(sections)
    for attempt in range(_MAX_REPAIR_ATTEMPTS):
        flagged = {
            i: issues
            for i, (s, planned) in enumerate(zip(current, batch))
            if (issues := _completeness_issues(s, planned))
        }
        if not flagged:
            break

        logger.info(
            "course_section_incomplete",
            extra={"attempt": attempt + 1, "sections": {batch[i].title: issues for i, issues in flagged.items()}},
        )
        targets = [batch[i] for i in flagged]
        requirements = "\n".join(f"- « {batch[i].title} » : {' ; '.join(issues)}" for i, issues in flagged.items())
        prompt = (
            f'mode="{mode}"\n'
            f"Question de l'utilisateur : {question}\n\n"
            f"Contexte source (figé lors de la planification) :\n{context_block}\n\n"
            f"--- Plan complet validé (ne développe PAS les sections hors lot) ---\n{outline}\n\n"
            f"--- Sections à RÉGÉNÉRER intégralement ---\n{_format_sections(targets, detailed=True)}\n\n"
            "Une première version de ces sections avait des lacunes :\n"
            f"{requirements}\n\n"
            f"Génère exactement {len(targets)} section(s) DEVELOPMENT, mêmes titres et même ordre : remplis "
            "TOUTES les sous-sections Pourquoi/Quoi/Comment (jamais vide), couvre en détail tous les sous-thèmes "
            "listés, inclus au moins un bloc visuel (TABLE, LIST, DIAGRAM, CHART ou FORMULA) et renseigne "
            "`covered_subtopics`."
        )
        try:
            structured = await gemini_client.format_structured(
                raw_answer=prompt,
                system_instruction=_get_teacher_instructions(),
                response_schema=SectionsBatchSchema,
            )
            replacements = _align_batch_sections(SectionsBatchSchema.model_validate(structured).sections, targets)
        except (GeminiServiceError, ValidationError) as exc:
            logger.warning("course_section_repair_failed", extra={"attempt": attempt + 1, "error": str(exc)})
            break

        updated = list(current)
        for (index, issues), new in zip(flagged.items(), replacements):
            if _has_content(new) and len(_completeness_issues(new, batch[index])) < len(issues):
                updated[index] = new
        current = updated
    return current


def _finalize_incomplete(sections: list[Section], batch: list[ApiPlannedSection]) -> list[Section]:
    """Filet de sécurité sans appel Gemini supplémentaire : toute section encore incomplète après
    la boucle de réparation bornée devient une section de repli visiblement incomplète (même
    convention que `_align_batch_sections` — voir `is_incomplete_section`), au lieu d'être livrée
    silencieusement avec des trous. L'apprenant la régénère lui-même (`POST .../regenerate`)."""
    return [
        section if not _completeness_issues(section, planned) else _incomplete_section(planned)
        for section, planned in zip(sections, batch)
    ]


async def _generate_wrap_up(
    planned: list[ApiPlannedSection],
    *,
    question: str,
    mode: str,
    context_block: str,
    meta: dict[str, Any],
    sources: list[Source],
    gemini_client: GeminiClient,
) -> CourseGenerationSchema | None:
    """Introduction, pièges, résumé, suite et quiz. Best-effort : None en cas d'échec."""
    development = [s for s in planned if s.type == "development"]
    wanted = [s for s in planned if s.type != "development"]
    prompt = (
        f'mode="{mode}"\n'
        f"Question de l'utilisateur : {question}\n\n"
        f"Contexte source (figé lors de la planification) :\n{context_block}\n\n"
        "--- Sections DEVELOPMENT du cours (déjà rédigées par ailleurs — ne les régénère pas) ---\n"
        f"{_format_sections(development, detailed=True)}\n\n"
        "--- Sections à générer maintenant ---\n"
        f"{_format_sections(wanted, detailed=True) or '(aucune section hors développement)'}\n\n"
        "Génère ces sections (mêmes titres, même ordre) ainsi que le quiz couvrant l'ensemble des sections "
        "DEVELOPMENT. N'inclus AUCUNE section de type development. Laisse `sources` vide. "
        "Renseigne `direct_answer` : la réponse directe à la question (2-3 phrases, points clés, 1 visuel "
        "récapitulatif), distincte de l'introduction dont elle ne reprend aucun texte. "
        "Propose dans `video_search_queries` 1 à 2 requêtes de recherche YouTube courtes (jamais une URL). "
        "Retourne le JSON selon le schéma fourni."
    )
    try:
        structured = await gemini_client.format_structured(
            raw_answer=prompt,
            system_instruction=_get_teacher_instructions(),
            response_schema=CourseGenerationSchema,
        )
        structured = _coerce_known_format(dict(structured), mode)
        # Champs dérivés du plan/contexte : écrasés côté code plutôt que confiés au modèle.
        structured["meta"] = meta
        structured["sources"] = [s.model_dump(mode="json") for s in sources]
        structured = await ensure_distinct_direct_answer(
            structured, mode=mode, gemini_client=gemini_client, system_instruction=_get_teacher_instructions(),
            context_prompt=prompt, max_overlap=get_settings().course_answer_intro_similarity_max,
        )
        try:
            return CourseGenerationSchema.model_validate(structured)
        except ValidationError as exc:
            if not _is_quiz_difficulty_error(exc):
                raise
            return CourseGenerationSchema.model_validate(_rebalance_quiz_difficulty(structured))
    except (GeminiServiceError, ValidationError, AttributeError, TypeError) as exc:
        logger.warning("course_plan_wrap_up_failed", extra={"error": str(exc)})
        return None


async def generate_course_from_validated_plan(
    plan_row: CoursePlan,
    edited_sections: list[ApiPlannedSection],
    gemini_client: GeminiClient,
    settings: Settings,
    db_session_factory: async_sessionmaker | None = None,
) -> CourseGenerationResponse:
    """Génère le cours complet à partir d'un plan validé, par lots de sections DEVELOPMENT.

    Paramètres:
        plan_row: plan persisté (question, mode, contexte figé, meta).
        edited_sections: sections du plan telles que validées/éditées par l'utilisateur
            (source de vérité de la structure ; revalidées par Pydantic en amont).
        gemini_client: client Gemini.
        settings: configuration (`course_plan_batch_size`).

    Retour: CourseGenerationResponse (contrat identique à `/courses/generate`).

    Fonctionnement: N lots de `course_plan_batch_size` sections DEVELOPMENT
    (séquentiels — respecte le rate limit Gemini), puis 1 appel pour
    introduction / pièges / résumé / suite / quiz, fusionnés dans l'ordre du
    plan et mappés par le pipeline existant (`_validate_and_map`). Aucun plafond
    de sections ; pas de contrôle de couverture post-génération (le plan fait foi).

    Résilience: un lot en échec devient des sections marquées « incomplètes » sans
    bloquer les autres ; si TOUS les lots échouent, l'erreur est propagée (502/503).

    Lève: GeminiInvalidResponseError (tous les lots en échec ou mapping impossible).
    """
    mode = plan_row.mode
    question = plan_row.question
    retrieval_context = plan_row.retrieval_context
    context_block = _render_context(retrieval_context)
    sources = _sources_from_context(retrieval_context)

    sections = [
        s.model_copy(update={"order": i})
        for i, s in enumerate(sorted(edited_sections, key=lambda s: s.order), start=1)
    ]
    outline = _format_sections(sections, detailed=False)
    development = [s for s in sections if s.type == "development"]
    batches = _chunked(development, settings.course_plan_batch_size)

    generated: list[Section] = []
    last_error: Exception | None = None
    failed_batches = 0
    for number, batch in enumerate(batches, start=1):
        try:
            generated.extend(
                await _generate_batch(
                    batch, question=question, mode=mode, context_block=context_block,
                    outline=outline, gemini_client=gemini_client,
                )
            )
        except (GeminiServiceError, ValidationError) as exc:
            last_error = exc
            failed_batches += 1
            logger.warning(
                "course_plan_batch_failed",
                extra={"batch": number, "batches": len(batches), "error": str(exc)},
            )
            generated.extend(_incomplete_section(s) for s in batch)

    if failed_batches == len(batches):
        raise GeminiInvalidResponseError(
            f"Génération impossible : tous les lots de sections ont échoué ({last_error})"
        ) from last_error

    plan_meta = plan_row.plan["meta"]
    meta = {
        "title": plan_meta["title"],
        "subject": plan_meta["subject"],
        "language": plan_meta.get("language", "fr"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    wrap_up = await _generate_wrap_up(
        sections, question=question, mode=mode, context_block=context_block,
        meta=meta, sources=sources, gemini_client=gemini_client,
    )

    wrap_sections = _align_wrap_up_titles(
        [s for s in (wrap_up.sections if wrap_up else []) if s.type is not SectionType.DEVELOPMENT],
        [s for s in sections if s.type != "development"],
    )
    ordered_sections = (
        [s for s in wrap_sections if s.type is SectionType.INTRODUCTION]
        + generated
        + [s for s in wrap_sections if s.type is not SectionType.INTRODUCTION]
    )
    structured = {
        "mode": mode,
        "format": "focused_answer",
        "meta": meta,
        "sources": [s.model_dump(mode="json") for s in sources],
        "sections": [s.model_dump(mode="json") for s in ordered_sections],
        "direct_answer": wrap_up.direct_answer.model_dump(mode="json") if wrap_up and wrap_up.direct_answer else None,
        "quiz": [q.model_dump(mode="json") for q in wrap_up.quiz] if wrap_up else [],
        "confidence": wrap_up.confidence.value if wrap_up else "medium",
        "unconfirmed_points": wrap_up.unconfirmed_points if wrap_up else [_WRAP_UP_FAILED_NOTE],
        "video_search_queries": wrap_up.video_search_queries if wrap_up else [],
    }
    with_videos = await attach_verified_videos(
        _validate_and_map(structured, mode), settings, gemini_client,
        search_queries=structured["video_search_queries"], db_session_factory=db_session_factory,
    )
    return await resolve_visuals(
        with_videos, settings=settings, db_session_factory=db_session_factory, gemini_client=gemini_client,
    )


# ─── Assistance IA sur le plan : compléter une section / ajouter des sections ────

# Extraits ciblés récupérés par fichier lors de la complétion d'une section.
_REFINE_TOP_K_PER_FILE = 4
# Plafond de sécurité du nombre de sections créées par « Ajouter plus de sections ».
_MORE_SECTIONS_MAX = 6


def _clip(text: str, limit: int) -> str:
    return text.strip()[:limit].strip()


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        cleaned = _clip(item, _PLAN_SUBTOPIC_MAX)
        if cleaned and cleaned.casefold() not in seen:
            seen.add(cleaned.casefold())
            result.append(cleaned)
    return result[:_PLAN_SUBTOPICS_MAX_ITEMS]


def _to_api_section(section: PlannedSection, *, type_: str, order: int, fallback_title: str = "") -> ApiPlannedSection:
    """Convertit une section produite par Gemini en section d'API, bornée aux limites du contrat."""
    return ApiPlannedSection(
        type=type_,
        title=_clip(section.title, _PLAN_TITLE_MAX) or fallback_title,
        objective=_clip(section.objective, _PLAN_OBJECTIVE_MAX),
        subtopics=_dedupe(section.subtopics),
        order=order,
    )


async def _targeted_context(
    plan_row: CoursePlan,
    section: ApiPlannedSection,
    *,
    vector_store: NumpyVectorStore,
    gemini_client: GeminiClient,
    settings: Settings,
) -> str:
    """Contexte figé du plan + recherche complémentaire ciblée sur la section.

    Fichiers : extraits les plus proches du titre/objectif de la section.
    Question seule avec grounding : recherche web ciblée. Best-effort : en cas
    d'échec de la recherche, on garde le contexte figé (la complétion reste possible).
    """
    base = _render_context(plan_row.retrieval_context)
    topic = f"{section.title}. {section.objective}".strip()
    extras: list[str] = []
    try:
        if plan_row.mode == "question_only":
            if settings.gemini_use_search_grounding:
                raw_answer, _ = await gemini_client.search_grounded(
                    prompt=(
                        "Recherche les informations précises, mécanismes, formules et cas d'usage à "
                        f"couvrir pour la section de cours suivante.\nSujet du cours : {plan_row.question}\n"
                        f"Section : {topic}"
                    ),
                    system_instruction=_get_plan_instructions(),
                )
                extras.append(f"Recherche complémentaire ciblée :\n{raw_answer}")
        else:
            chunks: list[dict[str, Any]] = []
            for filename in list(plan_row.filenames) or [None]:
                chunks.extend(
                    await vector_store.search(topic, top_k=_REFINE_TOP_K_PER_FILE, filename_filter=filename)
                )
            if chunks:
                extras.append(f"Extraits complémentaires ciblés :\n{_build_context_block(chunks)}")
    except GeminiServiceError as exc:
        logger.warning("course_plan_refine_search_failed", extra={"error": str(exc)})
    return "\n\n".join([base, *extras])


def _user_request_clause(instructions: str | None) -> str:
    if instructions:
        return (
            "Demande de l'utilisateur — informations à ajouter à cette section (à intégrer en priorité) :\n"
            f"{instructions}"
        )
    return (
        "L'utilisateur n'a rien précisé : détermine toi-même, à partir du contexte, ce qui manque à cette "
        "section (notions, mécanismes, formules, cas) et ajoute-le, sans rien lui demander."
    )


async def refine_planned_section(
    plan_row: CoursePlan,
    section: ApiPlannedSection,
    outline: list[ApiPlannedSection],
    instructions: str | None,
    vector_store: NumpyVectorStore,
    gemini_client: GeminiClient,
    settings: Settings,
) -> ApiPlannedSection:
    """Complète une section de plan jugée incomplète (titre, objectif, sous-thèmes).

    Paramètres:
        plan_row: plan persisté (question, mode, contexte figé, fichiers).
        section: section actuelle, telle qu'éditée par l'utilisateur (l'« ancienne » section).
        outline: plan complet courant, pour éviter les doublons avec les autres sections.
        instructions: précisions de l'utilisateur sur ce qu'il faut ajouter (None = automatique).

    Retour: la section enrichie ; `type` et `order` sont conservés à l'identique.

    Fonctionnement: contexte figé + recherche ciblée (extraits fichiers ou recherche web),
    puis 1 appel `format_structured`. Le contenu existant est conservé, les éléments
    manquants sont ajoutés.

    Lève: GeminiUnavailableError, GeminiQuotaExceededError, GeminiInvalidResponseError.
    """
    context_block = await _targeted_context(
        plan_row, section, vector_store=vector_store, gemini_client=gemini_client, settings=settings
    )
    others = [s for s in outline if s.order != section.order]
    prompt = (
        f'mode="{plan_row.mode}"\n'
        f"Question de l'utilisateur : {plan_row.question}\n\n"
        f"Contexte :\n{context_block}\n\n"
        "--- Autres sections du plan (pour éviter les doublons) ---\n"
        f"{_format_sections(others, detailed=False) or '(aucune)'}\n\n"
        "--- Section à compléter (version actuelle, jugée incomplète) ---\n"
        f"{_format_section(section, detailed=True)}\n\n"
        f"{_user_request_clause(instructions)}\n\n"
        "Retourne UNE seule section, en JSON selon le schéma fourni : conserve tout ce qui est correct "
        "dans la version actuelle, précise l'objectif si besoin et ajoute les sous-thèmes manquants "
        "(structure uniquement, aucun contenu rédigé). Garde le même rôle et, sauf titre trop vague, le "
        "même titre. Ne recopie pas le contenu des autres sections."
    )
    structured = await gemini_client.format_structured(
        raw_answer=prompt,
        system_instruction=_get_plan_instructions(),
        response_schema=PlannedSection,
    )
    try:
        refined = PlannedSection.model_validate(structured)
        result = _to_api_section(refined, type_=section.type, order=section.order, fallback_title=section.title)
    except ValidationError as exc:
        raise GeminiInvalidResponseError(f"Section complétée invalide: {exc}") from exc

    if not result.subtopics:
        result = result.model_copy(update={"subtopics": section.subtopics})
    return result


class _LenientMoreSections(MoreSectionsSchema):
    """Exigé par le schéma Gemini, mais son absence n'invalide pas les sections créées (relance ensuite)."""

    next_steps: PlannedSection | None = None


@dataclass(frozen=True)
class MoreSectionsResult:
    sections: list[ApiPlannedSection]
    next_steps: ApiPlannedSection | None = None


def _is_covered(topic: str, covered: set[str]) -> bool:
    key = _norm(topic)
    return not key or any(key == c or key in c or c in key for c in covered if c)


def _fresh_next_steps(
    candidate: PlannedSection | None,
    existing: ApiPlannedSection,
    current_sections: list[ApiPlannedSection],
    created: list[ApiPlannedSection],
) -> ApiPlannedSection | None:
    """Section « Pour aller plus loin » du modèle privée des pistes déjà couvertes ; None si aucune piste inédite."""
    if candidate is None:
        return None
    covered = {_norm(s.title) for s in [*current_sections, *created]}
    covered |= {_norm(topic) for topic in existing.subtopics}
    refreshed = _to_api_section(candidate, type_="next_steps", order=existing.order, fallback_title=existing.title)
    fresh = [topic for topic in refreshed.subtopics if not _is_covered(topic, covered)]
    if not fresh:
        return None
    return refreshed.model_copy(
        update={
            "title": existing.title,  # le titre a pu être édité par l'utilisateur
            "objective": refreshed.objective or existing.objective,
            "subtopics": fresh,
        }
    )


def _without_developed(existing: ApiPlannedSection, created: list[ApiPlannedSection]) -> ApiPlannedSection:
    """Dernier recours : l'ancienne section privée des pistes qui viennent de devenir des sections."""
    developed = {_norm(s.title) for s in created}
    remaining = [t for t in existing.subtopics if not _is_covered(t, developed)]
    return existing.model_copy(update={"subtopics": remaining})


async def _refreshed_next_steps(
    candidate: PlannedSection | None,
    current_sections: list[ApiPlannedSection],
    created: list[ApiPlannedSection],
    *,
    question: str,
    gemini_client: GeminiClient,
) -> ApiPlannedSection | None:
    """« Pour aller plus loin » actualisée : nouvelles pistes uniquement, titre et position conservés.

    None uniquement si le plan n'a pas de section next_steps. Sinon : pistes du modèle, à défaut
    une relance ciblée, à défaut l'ancienne section privée des pistes développées (jamais inchangée).
    """
    existing = next((s for s in current_sections if s.type == "next_steps"), None)
    if existing is None:
        return None

    refreshed = _fresh_next_steps(candidate, existing, current_sections, created)
    if refreshed is None:
        logger.info("course_plan_next_steps_retry")
        try:
            retry = await gemini_client.format_structured(
                raw_answer=(
                    f"Question de l'utilisateur : {question}\n\n"
                    f"Plan courant :\n{_format_sections([*current_sections, *created], detailed=False)}\n\n"
                    "Propose 3 à 5 pistes « Pour aller plus loin » NOUVELLES uniquement : aucune ne doit reprendre "
                    "un sujet déjà présent dans ce plan. Retourne le JSON selon le schéma fourni."
                ),
                system_instruction=_get_plan_instructions(),
                response_schema=NextStepsSchema,
            )
            refreshed = _fresh_next_steps(
                NextStepsSchema.model_validate(structured_or_raise(retry)).next_steps,
                existing, current_sections, created,
            )
        except (GeminiServiceError, ValidationError, ValueError):
            logger.warning("course_plan_next_steps_retry_failed", exc_info=True)
    if refreshed is None:
        logger.info("course_plan_next_steps_fallback")
        refreshed = _without_developed(existing, created)
    return refreshed


def structured_or_raise(value: Any) -> Any:
    if not isinstance(value, dict):
        raise ValueError("réponse structurée invalide")
    return value


async def generate_more_sections(
    plan_row: CoursePlan,
    current_sections: list[ApiPlannedSection],
    gemini_client: GeminiClient,
) -> MoreSectionsResult:
    """Crée de nouvelles sections de développement à partir de « Pour aller plus loin ».

    Paramètres:
        plan_row: plan persisté (question, mode, contexte figé).
        current_sections: plan complet courant (tel qu'édité) ; la ou les sections
            `next_steps` fournissent les pistes à développer.

    Retour: `sections` = 1 à `_MORE_SECTIONS_MAX` sections `development` inédites (titres absents du plan),
    `order` à la suite du plan courant (le client choisit l'emplacement d'insertion) ; `next_steps` =
    la section « Pour aller plus loin » actualisée avec de nouvelles pistes (le client remplace
    l'ancienne : ses pistes viennent d'être transformées en sections), ou None.

    Lève: GeminiUnavailableError, GeminiQuotaExceededError, GeminiInvalidResponseError
    (y compris quand le modèle ne retourne aucune section inédite).
    """
    next_steps = [s for s in current_sections if s.type == "next_steps"]
    leads = _format_sections(next_steps, detailed=True) or (
        "(aucune section « Pour aller plus loin » : propose des prolongements naturels du sujet)"
    )
    prompt = (
        f'mode="{plan_row.mode}"\n'
        f"Question de l'utilisateur : {plan_row.question}\n\n"
        f"Contexte source (figé lors de la planification) :\n{_render_context(plan_row.retrieval_context)}\n\n"
        f"--- Plan actuel (ne PAS répéter ces sections) ---\n{_format_sections(current_sections, detailed=False)}\n\n"
        f"--- Pistes « Pour aller plus loin » à développer ---\n{leads}\n\n"
        f"Crée de 3 à {_MORE_SECTIONS_MAX} NOUVELLES sections de type development qui développent ces pistes : "
        "titres thématiques précis (jamais génériques), objectif, 3 à 8 sous-thèmes chacune, ordonnées par "
        "dépendances logiques. Structure uniquement, aucun contenu rédigé. "
        "Mets ensuite à jour la section « Pour aller plus loin » (champ `next_steps`) : ces pistes viennent d'être "
        "transformées en sections, propose donc 3 à 5 NOUVELLES pistes qui prolongent le plan APRÈS ces sections, "
        "sans reprendre aucune piste ni aucun sujet déjà présent dans le plan. "
        "Retourne le JSON selon le schéma fourni."
    )
    structured = await gemini_client.format_structured(
        raw_answer=prompt,
        system_instruction=_get_plan_instructions(),
        response_schema=MoreSectionsSchema,
    )
    try:
        parsed = _LenientMoreSections.model_validate(structured)
        generated = parsed.planned_sections
    except ValidationError as exc:
        raise GeminiInvalidResponseError(f"Nouvelles sections invalides: {exc}") from exc

    known_titles = {s.title.strip().casefold() for s in current_sections}
    next_order = max((s.order for s in current_sections), default=0) + 1
    created: list[ApiPlannedSection] = []
    for candidate in sorted(generated, key=lambda s: s.order):
        title = _clip(candidate.title, _PLAN_TITLE_MAX)
        if not title or title.casefold() in known_titles or len(created) >= _MORE_SECTIONS_MAX:
            continue
        known_titles.add(title.casefold())
        created.append(_to_api_section(candidate, type_="development", order=next_order + len(created)))

    if not created:
        raise GeminiInvalidResponseError("Aucune nouvelle section exploitable n'a été générée")
    next_steps_section = await _refreshed_next_steps(
        parsed.next_steps, current_sections, created, question=plan_row.question, gemini_client=gemini_client
    )
    return MoreSectionsResult(sections=created, next_steps=next_steps_section)
