"""Accès à `flashcard_variants` (variantes générées des cartes de révision)."""

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import FlashcardVariant

VariantIndex = dict[tuple[uuid.UUID, str], dict[int, FlashcardVariant]]


async def get_for_sessions(session: AsyncSession, session_ids: list[uuid.UUID]) -> VariantIndex:
    """Variantes des sessions données, indexées par (session_id, card_id) puis par `variant_no`."""
    if not session_ids:
        return {}
    result = await session.execute(select(FlashcardVariant).where(FlashcardVariant.session_id.in_(session_ids)))
    index: VariantIndex = {}
    for variant in result.scalars().all():
        index.setdefault((variant.session_id, variant.card_id), {})[variant.variant_no] = variant
    return index


def add_variants(session: AsyncSession, variants: list[FlashcardVariant]) -> None:
    """Ajoute des variantes (sans commit : l'appelant contrôle la transaction)."""
    session.add_all(variants)
