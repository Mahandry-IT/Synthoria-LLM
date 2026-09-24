"""Normalisation des licences (Wikimedia Commons, Openverse) vers un code canonique.

Rejet par défaut : une licence absente ou non reconnue n'est jamais retenue (voir `is_allowed`) —
mieux vaut perdre une image que d'afficher un contenu dont la licence réelle est incertaine.
"""

from __future__ import annotations

import html
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.core.config import Settings

_TAG_RE = re.compile(r"<[^>]+>")

# Ordre significatif : les variantes NC/ND (toujours rejetées par la liste blanche par défaut)
# doivent être reconnues avant leurs sous-chaînes moins spécifiques ("by-sa", "by") pour ne pas
# les classer par erreur comme une licence permissive.
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bcc[\s-]*0\b"), "cc0"),
    (re.compile(r"public\s*domain|\bpdm\b|^pd[\s-]|\bpd[\s-]old\b"), "pdm"),
    (re.compile(r"by[\s-]*nc[\s-]*sa\b"), "by-nc-sa"),
    (re.compile(r"by[\s-]*nc[\s-]*nd\b"), "by-nc-nd"),
    (re.compile(r"by[\s-]*nc\b"), "by-nc"),
    (re.compile(r"by[\s-]*nd\b"), "by-nd"),
    (re.compile(r"by[\s-]*sa\b"), "by-sa"),
    (re.compile(r"\bby\b"), "by"),
]


def normalize_license(raw: str | None) -> str | None:
    """Code canonique (`cc0`, `pdm`, `by`, `by-sa`, `by-nc`, `by-nc-sa`, `by-nc-nd`, `by-nd`) ou
    None si `raw` est vide ou ne correspond à aucune licence reconnue."""
    if not raw:
        return None
    text = raw.strip().lower()
    for pattern, code in _PATTERNS:
        if pattern.search(text):
            return code
    return None


def is_allowed(code: str | None, settings: "Settings") -> bool:
    """Une licence absente ou hors liste blanche (ND/NC par défaut) n'est jamais retenue."""
    return code is not None and code in settings.media_web_allowed_licenses


_CANONICAL_URLS = {
    "by": "https://creativecommons.org/licenses/by/4.0/",
    "by-sa": "https://creativecommons.org/licenses/by-sa/4.0/",
    "by-nc": "https://creativecommons.org/licenses/by-nc/4.0/",
    "by-nd": "https://creativecommons.org/licenses/by-nd/4.0/",
    "by-nc-sa": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    "by-nc-nd": "https://creativecommons.org/licenses/by-nc-nd/4.0/",
    "cc0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "pdm": "https://creativecommons.org/publicdomain/mark/1.0/",
}


def canonical_url(code: str | None) -> str | None:
    """URL générique de la licence — repli quand le fournisseur n'en donne pas."""
    return _CANONICAL_URLS.get(code) if code else None


def strip_html(text: str | None) -> str:
    """Retire les balises HTML et décode les entités — le champ `Artist` de Commons en contient."""
    if not text:
        return ""
    return html.unescape(_TAG_RE.sub("", text)).strip()
