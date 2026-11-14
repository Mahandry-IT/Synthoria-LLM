"""Ajoute du contenu (une ou plusieurs sections) à un cours déjà persisté.

Le sujet des nouvelles sections vient, par ordre de priorité :
1. `instructions` fournies par l'apprenant (éditeur riche, converties en texte simple) ;
2. à défaut, les pistes de la section « Prochaines étapes » (`next_steps`) du cours, si présentes ;
3. à défaut, de nouveaux sujets proposés par le modèle à partir du cours existant.

Comme `section_regenerator`, toujours un appel Gemini bloquant (jamais de repli silencieux) :
l'apprenant doit savoir si sa demande n'a pas abouti plutôt que de se voir renvoyer une liste vide.
"""

from pydantic import ValidationError

from app.api.schemas import ApiPlannedSection
from app.core.config import Settings
from app.core.exceptions import GeminiInvalidResponseError
from app.db.models import CourseSession
from app.schemas.course_generation import Section, SectionType
from app.services.course_depth import CYCLE_CONTENT_DEMAND, DepthProfile, depth_of_session, get_profile
from app.services.course_generator import _get_teacher_instructions, session_context_block
from app.services.course_plan_generator import (
    _finalize_incomplete,
    _repair_incomplete_sections,
    batch_rules,
    batch_schema,
)
from app.services.gemini_client import GeminiClient
from app.services.vector_store import NumpyVectorStore


async def _validated_cycle_sections(
    sections: list[Section],
    row: CourseSession,
    context_block: str,
    gemini_client: GeminiClient,
    profile: DepthProfile,
    max_repairs: int,
) -> list[Section]:
    """Sections du format cycle : même validation de structure que la génération par lots, réparation bornée
    (`max_repairs` appels), puis repli « incomplète » (régénérable) pour toute section encore invalide, jamais une
    section au cycle vide ou absent livrée telle quelle."""
    batch = [
        ApiPlannedSection(type="development", title=s.title, order=i) for i, s in enumerate(sections, start=1)
    ]
    outline = "\n".join(f"- {s.title}" for s in sections)
    repaired = await _repair_incomplete_sections(
        sections, batch, question=row.question, mode=row.mode, context_block=context_block, outline=outline,
        gemini_client=gemini_client, profile=profile, max_attempts=max_repairs,
    )
    return _finalize_incomplete(repaired, batch, profile)


def _topic_clause(row: CourseSession, instructions: str) -> tuple[str, bool]:
    """Détermine la consigne de sujet donnée au modèle.

    Retour: (texte de la consigne, True si elle consomme les `next_steps` actuels — auquel cas
    ils doivent être vidés après génération, ayant tous été transformés en sections).
    """
    if instructions:
        return f"Sujet(s) à développer en nouvelle(s) section(s) :\n{instructions}", False

    next_steps = row.gemini_response.get("next_steps") or []
    if next_steps:
        pistes = "\n".join(f"- {step}" for step in next_steps)
        return f"Développe chacune des pistes « Prochaines étapes » suivantes en section(s) :\n{pistes}", True

    return (
        "Aucun sujet précisé : détermine toi-même, à partir du cours existant, un ou plusieurs "
        "sujets pertinents non encore couverts qui prolongent naturellement ce cours.",
        False,
    )


async def add_course_sections(
    row: CourseSession,
    instructions: str,
    gemini_client: GeminiClient,
    vector_store: NumpyVectorStore,
    settings: Settings,
) -> tuple[list[Section], list[str]]:
    """Génère de nouvelles sections DEVELOPMENT pour un cours déjà persisté.

    Retour: (nouvelles sections, `next_steps` finaux à persister après l'opération).

    Lève: GeminiUnavailableError, GeminiQuotaExceededError, GeminiInvalidResponseError.
    """
    context_block = await session_context_block(
        row.question, row.mode, row.filenames, vector_store, gemini_client, settings
    )
    existing_titles = [s.get("title", "") for s in (row.gemini_response.get("sections") or [])]
    topic_clause, consumes_next_steps = _topic_clause(row, instructions)
    # Les nouvelles sections suivent le mode du cours (`meta.depth`, absent = approfondi).
    profile = get_profile(depth_of_session(row.gemini_response))
    schema = batch_schema(profile)
    # Modes standard et approfondi : sections au format cycle (objet `cycle`) ; sinon Pourquoi / Quoi / Comment.
    content = (
        f"chacune avec un titre thématique précis (jamais générique) et {CYCLE_CONTENT_DEMAND}"
        if profile.use_cycle
        else "chacune avec un titre thématique précis (jamais générique), ses sous-sections Pourquoi/Quoi/Comment "
        "(toutes obligatoires, avec un exemple travaillé complet dans Comment), un défi (`challenge`), un exemple "
        "à trous (`faded_example`), les `check_questions`, un `recall_prompt`, les `key_takeaways` et les "
        "`pitfalls_to_avoid`"
    )

    prompt = (
        f"Question d'origine du cours : {row.question}\n\n"
        f"Contexte source :\n{context_block}\n\n"
        "--- Sections déjà présentes dans ce cours (ne PAS les répéter) ---\n"
        + ("\n".join(f"- {t}" for t in existing_titles) or "(aucune)")
        + f"\n\n{topic_clause}\n\n"
        f"{batch_rules(profile)}\n\n"
        f"Génère une ou plusieurs sections DEVELOPMENT complètes, {content}. Retourne le JSON selon le schéma fourni."
    )
    structured = await gemini_client.format_structured(
        raw_answer=prompt, system_instruction=_get_teacher_instructions(), response_schema=schema,
    )
    try:
        parsed = schema.model_validate(structured)
    except ValidationError as exc:
        raise GeminiInvalidResponseError(f"Nouvelles sections invalides: {exc}") from exc
    if not parsed.sections:
        raise GeminiInvalidResponseError("Aucune nouvelle section n'a été générée")

    sections = [s.model_copy(update={"type": SectionType.DEVELOPMENT}) for s in parsed.sections]
    if profile.use_cycle:
        sections = await _validated_cycle_sections(
            sections, row, context_block, gemini_client, profile, max(0, settings.section_regenerate_max_repairs)
        )
    next_steps = [] if consumes_next_steps else list(row.gemini_response.get("next_steps") or [])
    return sections, next_steps
