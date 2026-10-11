"""Régénère le contenu d'UNE section « incomplète » d'un cours déjà persisté.

Contrairement à la génération complète du cours (best-effort, replis silencieux), cette
régénération ciblée lève toujours en cas d'échec du premier appel : l'apprenant doit savoir que sa
tentative n'a pas abouti plutôt que de se voir renvoyer silencieusement l'ancienne section
incomplète. La section obtenue passe ensuite par la même validation que la génération depuis un
plan (sous-sections, ratio non textuel, budget du mode) et une réparation bornée par
`section_regenerate_max_repairs` (1 par défaut, contre 2 pour la génération complète).
"""

import logging

from pydantic import ValidationError

from app.core.config import Settings
from app.core.exceptions import GeminiInvalidResponseError, GeminiServiceError
from app.db.models import CourseSession
from app.schemas.course_generation import Section, SectionType, SectionsBatchSchema
from app.services.course_depth import DepthProfile, depth_of_session, get_profile, render_rules
from app.services.course_generator import _get_teacher_instructions, session_context_block
from app.services.course_plan_generator import is_better_replacement, section_issues
from app.services.gemini_client import GeminiClient
from app.services.vector_store import NumpyVectorStore

logger = logging.getLogger(__name__)


async def _generate_one(
    prompt: str, section_title: str, system_instruction: str, gemini_client: GeminiClient
) -> Section:
    structured = await gemini_client.format_structured(
        raw_answer=prompt, system_instruction=system_instruction, response_schema=SectionsBatchSchema,
    )
    try:
        parsed = SectionsBatchSchema.model_validate(structured)
    except ValidationError as exc:
        raise GeminiInvalidResponseError(f"Section régénérée invalide: {exc}") from exc
    if not parsed.sections:
        raise GeminiInvalidResponseError("Aucune section n'a été renvoyée")
    return parsed.sections[0].model_copy(update={"type": SectionType.DEVELOPMENT, "title": section_title})


async def _repair(
    section: Section,
    base_prompt: str,
    *,
    profile: DepthProfile,
    system_instruction: str,
    gemini_client: GeminiClient,
    max_repairs: int,
) -> Section:
    """Boucle de validation bornée (`max_repairs`) : garde la meilleure version (score pondéré) ;
    un échec Gemini pendant la réparation conserve la meilleure version obtenue."""
    best, best_issues = section, section_issues(section, None, profile)
    for attempt in range(max_repairs):
        if not best_issues:
            break
        logger.info(
            "course_section_regenerate_repair",
            extra={"attempt": attempt + 1, "section": section.title, "issues": best_issues.all},
        )
        prompt = (
            f"{base_prompt}\n\n"
            "Une première version de cette section avait les problèmes suivants, à corriger :\n"
            + "\n".join(f"- {issue}" for issue in best_issues.all)
        )
        try:
            candidate = await _generate_one(prompt, section.title, system_instruction, gemini_client)
        except GeminiServiceError as exc:
            logger.warning("course_section_regenerate_repair_failed", extra={"attempt": attempt + 1, "error": str(exc)})
            break
        candidate_issues = section_issues(candidate, None, profile)
        if is_better_replacement(candidate, best_issues, candidate_issues):
            best, best_issues = candidate, candidate_issues

    if best_issues.blocking:
        logger.warning("course_section_regenerate_incomplete", extra={"section": section.title, "issues": best_issues.all})
    elif best_issues.soft:
        logger.warning(
            "course_section_budget_exceeded",
            extra={"section": section.title, "depth": profile.name, "issues": list(best_issues.soft)},
        )
    return best


async def regenerate_section(
    row: CourseSession,
    section_title: str,
    gemini_client: GeminiClient,
    vector_store: NumpyVectorStore,
    settings: Settings,
) -> Section:
    """Régénère une section DEVELOPMENT complète (cycle pédagogique inclus), par son titre.

    Le mode du cours (`meta.depth` de la session, absent = approfondi) fixe les règles de forme
    injectées dans le prompt et vérifiées ensuite (`section_issues`, réparation bornée).

    Lève: GeminiUnavailableError, GeminiQuotaExceededError, GeminiInvalidResponseError.
    """
    system_instruction = _get_teacher_instructions()
    profile = get_profile(depth_of_session(getattr(row, "gemini_response", None)))
    context_block = await session_context_block(
        row.question, row.mode, row.filenames, vector_store, gemini_client, settings
    )

    prompt = (
        f"Question de l'utilisateur (contexte du cours) : {row.question}\n\n"
        f"Contexte source :\n{context_block}\n\n"
        f"{render_rules(profile)}\n\n"
        f"Génère UNE SEULE section DEVELOPMENT intitulée exactement « {section_title} », complète : "
        "sous-sections Pourquoi/Quoi/Comment (avec un exemple travaillé complet dans Comment), un défi "
        "(`challenge`), un exemple à trous (`faded_example`), les `check_questions` et un `recall_prompt`. "
        "Retourne le JSON selon le schéma fourni."
    )
    section = await _generate_one(prompt, section_title, system_instruction, gemini_client)
    # Régénération déclenchée par l'utilisateur : borne de réparation plus basse que la génération
    # complète, chaque tentative supplémentaire étant un appel Gemini qu'il attend (et un risque de 429).
    return await _repair(
        section, prompt, profile=profile, system_instruction=system_instruction, gemini_client=gemini_client,
        max_repairs=max(0, settings.section_regenerate_max_repairs),
    )
