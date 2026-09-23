"""Saisies utilisateur en Markdown (éditeur riche du frontend) : conversion en texte brut.

Le Markdown est conservé tel quel dans les prompts Gemini (la mise en forme porte l'intention de
l'apprenant) ; le texte brut sert aux usages où la syntaxe serait du bruit : comparaisons de
sous-thèmes, titres affichés ou lus à voix haute.
"""

import re

_FENCE_RE = re.compile(r"^\s*(```|~~~)[^\n]*$", re.MULTILINE)
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_QUOTE_RE = re.compile(r"^\s*>\s?", re.MULTILINE)
_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(?:\[[ xX]\]\s+)?", re.MULTILINE)
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")
# Marqueurs d'emphase : on ne retire `_` qu'en bord de mot (x_1 et snake_case restent intacts).
_BOLD_RE = re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1")
_STAR_ITALIC_RE = re.compile(r"(?<![\w*])\*(?=\S)(.+?)(?<=\S)\*(?![\w*])")
_UNDERSCORE_ITALIC_RE = re.compile(r"(?<!\w)_(?=\S)(.+?)(?<=\S)_(?!\w)")
_STRIKE_RE = re.compile(r"~~(?=\S)(.+?)(?<=\S)~~")
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_ESCAPE_RE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!~|>])")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def markdown_to_plain(text: str) -> str:
    """Texte brut lisible d'une saisie Markdown (emphases, code, listes, liens retirés).

    Les formules `$...$` et les retours à la ligne sont conservés ; idempotent sur du texte brut.
    """
    if not text:
        return ""
    plain = _FENCE_RE.sub("", text)
    plain = _HEADING_RE.sub("", plain)
    plain = _QUOTE_RE.sub("", plain)
    plain = _LIST_MARKER_RE.sub("", plain)
    plain = _IMAGE_RE.sub(r"\1", plain)
    plain = _LINK_RE.sub(r"\1", plain)
    plain = _HTML_TAG_RE.sub("", plain)
    plain = _INLINE_CODE_RE.sub(r"\1", plain)
    for pattern in (_BOLD_RE, _STRIKE_RE, _STAR_ITALIC_RE, _UNDERSCORE_ITALIC_RE):
        plain = pattern.sub(lambda m: m.group(m.lastindex or 1), plain)
    plain = _ESCAPE_RE.sub(r"\1", plain)
    plain = plain.replace("&nbsp;", " ")
    lines = [line.rstrip() for line in plain.splitlines()]
    return _BLANK_LINES_RE.sub("\n\n", "\n".join(lines)).strip()
