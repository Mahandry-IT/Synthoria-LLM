"""Ajoute du contenu (une ou plusieurs sections) à un cours déjà persisté.

Le sujet des nouvelles sections vient, par ordre de priorité :
1. `instructions` fournies par l'apprenant (éditeur riche, converties en texte simple) ;
2. à défaut, les pistes de la section « Prochaines étapes » (`next_steps`) du cours, si présentes ;
3. à défaut, de nouveaux sujets proposés par le modèle à partir du cours existant.

Comme `section_regenerator`, toujours un appel Gemini bloquant (jamais de repli silencieux) :
l'apprenant doit savoir si sa demande n'a pas abouti plutôt que de se voir renvoyer une liste vide.
"""

from pydantic import ValidationError

from app.core.config import Settings
from app.core.exceptions import GeminiInvalidResponseError
from app.db.models import CourseSession
from app.schemas.course_generation import Section, SectionType, SectionsBatchSchema
from app.services.course_depth import depth_of_session, get_profile, render_rules
from app.services.course_generator import _get_teacher_instructions, session_context_block
from app.services.gemini_client import GeminiClient
from app.services.vector_store import NumpyVectorStore


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
    rules = render_rules(get_profile(depth_of_session(row.gemini_response)))

    prompt = (
        f"Question d'origine du cours : {row.question}\n\n"
        f"Contexte source :\n{context_block}\n\n"
        "--- Sections déjà présentes dans ce cours (ne PAS les répéter) ---\n"
        + ("\n".join(f"- {t}" for t in existing_titles) or "(aucune)")
        + f"\n\n{topic_clause}\n\n"
        f"{rules}\n\n"
        "Génère une ou plusieurs sections DEVELOPMENT complètes, chacune avec un titre thématique "
        "précis (jamais générique), ses sous-sections Pourquoi/Quoi/Comment (toutes obligatoires, "
        "avec un exemple travaillé complet dans Comment), un défi (`challenge`), un exemple à trous "
        "(`faded_example`), les `check_questions` et un `recall_prompt`. Retourne le JSON selon le "
        "schéma fourni."
    )
    structured = await gemini_client.format_structured(
        raw_answer=prompt, system_instruction=_get_teacher_instructions(), response_schema=SectionsBatchSchema,
    )
    try:
        parsed = SectionsBatchSchema.model_validate(structured)
    except ValidationError as exc:
        raise GeminiInvalidResponseError(f"Nouvelles sections invalides: {exc}") from exc
    if not parsed.sections:
        raise GeminiInvalidResponseError("Aucune nouvelle section n'a été générée")

    sections = [s.model_copy(update={"type": SectionType.DEVELOPMENT}) for s in parsed.sections]
    next_steps = [] if consumes_next_steps else list(row.gemini_response.get("next_steps") or [])
    return sections, next_steps
