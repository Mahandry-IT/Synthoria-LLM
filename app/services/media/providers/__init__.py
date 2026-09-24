"""Fournisseurs d'images web (Wikimedia Commons, Openverse) : un seul contrat commun.

Chaque fournisseur ne fait QUE l'appel réseau + le mapping vers `WebImageCandidate` — aucun
filtrage ni décision métier (licence, taille, pertinence), qui vivent dans `visual_resolver.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class WebImageCandidate:
    """Champs bruts d'un résultat de recherche — non filtrés, non interprétés."""

    provider: str
    thumb_url: str
    download_url: str
    page_url: str
    width: int
    height: int
    title: str
    author: str | None
    license: str | None  # code brut du fournisseur ; normalisé ensuite par app.services.media.licenses
    license_url: str | None


class ImageProvider(Protocol):
    name: str

    async def search(self, query: str, *, limit: int) -> list[WebImageCandidate]: ...
