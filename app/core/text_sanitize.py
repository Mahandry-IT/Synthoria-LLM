"""Nettoyage des caractères de contrôle avant persistance en base.

Postgres refuse tout `\\u0000` (et plus largement, tout caractère de contrôle bas) dans une
colonne `text`/`jsonb` (`UntranslatableCharacterError`). Ces octets peuvent provenir d'une
extraction PDF (police/cmap corrompue) mais aussi d'une réponse LLM qui recopie du contenu source
mal décodé : le nettoyage doit donc s'appliquer aussi bien au texte brut qu'aux structures
JSON complètes (chunks de contexte, plan généré) juste avant l'écriture en base.
"""

import re
from typing import Any

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def strip_control_chars(text: str) -> str:
    return _CONTROL_CHARS_RE.sub("", text)


def sanitize_json(value: Any) -> Any:
    """Applique `strip_control_chars` récursivement à toute chaîne d'une structure JSON
    (dict/list imbriqués). Les autres types (int, bool, None, UUID...) sont retournés tels quels."""
    if isinstance(value, str):
        return strip_control_chars(value)
    if isinstance(value, dict):
        return {key: sanitize_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json(item) for item in value]
    return value
