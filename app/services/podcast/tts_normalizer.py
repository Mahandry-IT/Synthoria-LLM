"""Oralisation déterministe du texte avant synthèse vocale.

Fonctions pures : aucune E/S. Le but est qu'aucun symbole, formule, code ou
balisage ne soit lu tel quel par le moteur TTS.
"""

import re
import unicodedata

# Sigles épelés lettre par lettre (les autres, comme « ISO », restent prononcés normalement).
DEFAULT_ACRONYMS: dict[str, str] = {
    "CO2e": "C O deux équivalent",
    "CO2": "C O deux",
    "PUE": "P U E",
    "WUE": "W U E",
    "CUE": "C U E",
    "ACV": "A C V",
    "DEEE": "D E E E",
    "RGESN": "R G E S N",
    "GR491": "G R quatre cent quatre-vingt-onze",
    "IT": "I T",
    "IA": "I A",
    "CPU": "C P U",
    "GPU": "G P U",
    "RAM": "rame",
    "API": "A P I",
    "TTS": "T T S",
    "KPI": "K P I",
    "LLM": "L L M",
    "PDF": "P D F",
    "IP": "I P",
}

_UNITS: dict[str, str] = {
    "kWh": "kilowattheures",
    "MWh": "mégawattheures",
    "kW": "kilowatts",
    "MW": "mégawatts",
    "Go": "gigaoctets",
    "Mo": "mégaoctets",
    "Ko": "kilooctets",
    "To": "téraoctets",
    "Gb": "gigabits",
    "Mb": "mégabits",
    "ms": "millisecondes",
    "km": "kilomètres",
    "kg": "kilogrammes",
    "°C": "degrés Celsius",
    "°": "degrés",
}

_LATEX_WORDS: dict[str, str] = {
    r"\alpha": "alpha", r"\beta": "bêta", r"\gamma": "gamma", r"\delta": "delta",
    r"\epsilon": "epsilon", r"\varepsilon": "epsilon", r"\lambda": "lambda", r"\mu": "mu",
    r"\sigma": "sigma", r"\theta": "thêta", r"\pi": "pi", r"\rho": "rho",
    r"\sum": "somme de", r"\times": "fois", r"\cdot": "fois", r"\approx": "environ égal à",
    r"\leq": "inférieur ou égal à", r"\geq": "supérieur ou égal à", r"\neq": "différent de",
    r"\pm": "plus ou moins", r"\infty": "l'infini", r"\hat": "estimé de", r"\bar": "moyenne de",
    r"\text": "", r"\mathrm": "", r"\left": "", r"\right": "", r"\dots": "et ainsi de suite",
    r"\ldots": "et ainsi de suite",
}

_SYMBOLS: list[tuple[str, str]] = [
    ("≈", " environ égal à "), ("≤", " inférieur ou égal à "), ("≥", " supérieur ou égal à "),
    ("≠", " différent de "), ("×", " fois "), ("→", " donne "), ("⇒", " donne "),
    ("±", " plus ou moins "), ("÷", " divisé par "), ("−", " moins "), ("²", " au carré"),
    ("³", " au cube"),
]

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_FENCED_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`]*)`")
_DISPLAY_MATH_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_INLINE_MATH_RE = re.compile(r"\$([^$\n]+?)\$")
_LINK_RE = re.compile(r"\[([^\]]+)\]\((?:[^)]+)\)")
_CIDR_RE = re.compile(r"\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})/(\d{1,2})\b")
_IPV4_RE = re.compile(r"\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b")
_DECIMAL_RE = re.compile(r"(?<![\d.])(\d+)\.(\d+)(?![\d.])")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?…])\s+")


def latex_to_words(latex: str) -> str:
    """Traduit une expression LaTeX simple en texte prononçable."""
    text = latex
    text = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r" \1 sur \2 ", text)
    text = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r" racine de \1 ", text)
    text = re.sub(r"\^\s*\{?2\}?", " au carré ", text)
    text = re.sub(r"\^\s*\{?3\}?", " au cube ", text)
    text = re.sub(r"\^\s*\{([^{}]*)\}", r" puissance \1 ", text)
    text = re.sub(r"\^\s*(\w)", r" puissance \1 ", text)
    text = re.sub(r"_\s*\{([^{}]*)\}", r" indice \1 ", text)
    text = re.sub(r"_\s*(\w)", r" indice \1 ", text)
    for command in sorted(_LATEX_WORDS, key=len, reverse=True):
        text = text.replace(command, f" {_LATEX_WORDS[command]} ")
    text = re.sub(r"\\[A-Za-z]+", " ", text)  # commandes inconnues
    text = text.replace("\\", " ").replace("{", " ").replace("}", " ")
    return text


def _spell_ip(match: re.Match[str]) -> str:
    return " point ".join(match.groups())


def _spell_cidr(match: re.Match[str]) -> str:
    *octets, prefix = match.groups()
    return " point ".join(octets) + f" slash {prefix}"


def _replace_acronyms(text: str, acronyms: dict[str, str]) -> str:
    for sigle in sorted(acronyms, key=len, reverse=True):
        text = re.sub(rf"(?<![\w]){re.escape(sigle)}(?![\w])", acronyms[sigle], text)
    return text


def normalize_for_tts(text: str, *, acronyms: dict[str, str] | None = None) -> str:
    """Convertit un texte de cours en texte oral : plus de LaTeX, Markdown, code ni symboles."""
    acronyms = DEFAULT_ACRONYMS if acronyms is None else acronyms

    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_RE.sub(" ", text)

    # Code : un bloc n'est jamais lu ; le code en ligne garde ses mots.
    text = _FENCED_RE.sub(" un extrait de code est présenté dans le cours. ", text)
    text = _INLINE_CODE_RE.sub(r"\1", text)

    # Maths.
    text = _DISPLAY_MATH_RE.sub(lambda m: f" {latex_to_words(m.group(1))} ", text)
    text = _INLINE_MATH_RE.sub(lambda m: f" {latex_to_words(m.group(1))} ", text)

    # Markdown.
    text = _LINK_RE.sub(r"\1", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.MULTILINE)
    text = text.replace("**", "").replace("__", "").replace("*", " ")

    # Adresses réseau avant les décimaux.
    text = _CIDR_RE.sub(_spell_cidr, text)
    text = _IPV4_RE.sub(_spell_ip, text)
    text = _DECIMAL_RE.sub(r"\1,\2", text)

    # Unités et monnaies.
    text = re.sub(r"(\d)\s*€", r"\1 euros", text)
    text = text.replace("€", " euros ")
    text = re.sub(r"(\d)\s*%", r"\1 pour cent", text)
    text = text.replace("%", " pour cent ")
    for unit in sorted(_UNITS, key=len, reverse=True):
        pattern = rf"(?<=\d)\s?{re.escape(unit)}(?![\w])" if unit[0].isalpha() else rf"(?<=\d)\s?{re.escape(unit)}"
        text = re.sub(pattern, f" {_UNITS[unit]}", text)

    for symbol, spoken in _SYMBOLS:
        text = text.replace(symbol, spoken)
    text = re.sub(r"\s*=\s*", " égale ", text)
    text = re.sub(r"\s*<\s*", " inférieur à ", text)
    text = re.sub(r"\s*>\s*", " supérieur à ", text)
    text = re.sub(r"(?<=\d)\s*\+\s*(?=\d)", " plus ", text)
    text = re.sub(r"\bet/ou\b", "et ou", text)
    text = text.replace("&", " et ").replace("/", " sur ")

    text = _replace_acronyms(text, acronyms)

    text = re.sub(r"[\[\]{}|~^_#@\\]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def split_for_tts(text: str, max_chars: int = 400) -> list[str]:
    """Découpe un texte en morceaux d'au plus max_chars, en respectant les phrases."""
    if max_chars < 20:
        raise ValueError("max_chars trop petit")
    sentences = [s.strip() for s in _SENTENCE_END_RE.split(text.strip()) if s.strip()]

    pieces: list[str] = []
    for sentence in sentences:
        pieces.extend(_split_long_sentence(sentence, max_chars))

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current} {piece}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    return chunks


def _split_long_sentence(sentence: str, max_chars: int) -> list[str]:
    if len(sentence) <= max_chars:
        return [sentence]
    parts = [p.strip() for p in re.split(r"(?<=[,;:])\s+", sentence) if p.strip()]
    result: list[str] = []
    current = ""
    for part in parts:
        candidate = f"{current} {part}".strip()
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            result.append(current)
        current = ""
        if len(part) <= max_chars:
            current = part
        else:
            result.extend(_hard_split(part, max_chars))
    if current:
        result.append(current)
    return result


def _hard_split(text: str, max_chars: int) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    current = ""
    for word in words:
        while len(word) > max_chars:  # mot démesuré : coupe brute
            if current:
                chunks.append(current)
                current = ""
            chunks.append(word[:max_chars])
            word = word[max_chars:]
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = word
    if current:
        chunks.append(current)
    return chunks
