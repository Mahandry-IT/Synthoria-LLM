"""Rendu texte d'un cours persisté (`CourseSession.gemini_response`), injecté en contexte du chat.

Couvre tout le contenu lisible : introduction, réponse directe, sections (défi, sous-sections et
leurs blocs, exemple à compléter, questions « Vérifie »), pièges, synthèse et suites, avec repli
sur les champs hérités (`quoi`/`pourquoi`/`comment`, `worked_example`, `tables`) des anciennes
sessions sans sous-sections. Ne lève jamais : un champ absent ou mal typé est simplement ignoré.
"""

import math
import re
import unicodedata
from collections import Counter
from typing import Any

TRUNCATION_MARKER = "\n[… leçon tronquée …]"


def _s(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _items(value: Any) -> list:
    return value if isinstance(value, list) else []


def _dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _table_text(table: dict) -> str:
    lines = [_s(table.get("caption"))]
    headers = [str(h) for h in _items(table.get("headers"))]
    if headers:
        lines.append(" | ".join(headers))
    lines.extend(" | ".join(str(c) for c in _items(row)) for row in _items(table.get("rows")))
    return "\n".join(line for line in lines if line)


def _step_text(step: Any) -> str:
    """Étape d'exemple : chaîne (blocs) ou `{id, content}` (champ hérité de section)."""
    return _s(step.get("content")) if isinstance(step, dict) else _s(step)


def _worked_example_text(example: dict) -> str:
    steps = [_step_text(s) for s in _items(example.get("steps"))]
    lines = [f"Exemple : {_s(example.get('statement'))}" if _s(example.get("statement")) else ""]
    lines.extend(f"{i}. {s}" for i, s in enumerate((s for s in steps if s), start=1))
    if _s(example.get("result")):
        lines.append(f"Résultat : {_s(example.get('result'))}")
    return "\n".join(line for line in lines if line)


def _pitfall_text(pitfall: dict) -> str:
    parts = [
        ("Piège", _s(pitfall.get("description"))),
        ("Pourquoi", _s(pitfall.get("why_it_happens"))),
        ("Comment l'éviter", _s(pitfall.get("how_to_avoid"))),
    ]
    return "\n".join(f"{label} : {text}" for label, text in parts if text)


def _chart_text(chart: dict) -> str:
    labels = [str(label) for label in _items(chart.get("labels"))]
    lines = [f"Graphique : {_s(chart.get('caption'))}"]
    for series in _items(chart.get("series")):
        series = _dict(series)
        values = ", ".join(f"{label}={value}" for label, value in zip(labels, _items(series.get("values"))))
        lines.append(f"{_s(series.get('name'))} : {values}")
    return "\n".join(lines)


def block_text(block: dict) -> str:
    """Texte d'un bloc de contenu, quel que soit son type (vide si rien de lisible)."""
    block = _dict(block)
    parts: list[str] = [_s(block.get("text"))]
    ordered = bool(block.get("list_ordered"))
    for i, item in enumerate(_items(block.get("list_items")), start=1):
        parts.append(f"{i}. {_s(item)}" if ordered else f"- {_s(item)}")
    if block.get("table"):
        parts.append(_table_text(_dict(block["table"])))
    if block.get("formula"):
        formula = _dict(block["formula"])
        description = _s(formula.get("description"))
        parts.append(f"$${_s(formula.get('latex'))}$$" + (f" ({description})" if description else ""))
    if _s(block.get("code")):
        parts.append(f"```{_s(block.get('code_language'))}\n{block['code']}\n```")
    if block.get("worked_example"):
        parts.append(_worked_example_text(_dict(block["worked_example"])))
    if block.get("pitfall"):
        parts.append(_pitfall_text(_dict(block["pitfall"])))
    if _s(block.get("image_caption")):
        parts.append(f"Image : {_s(block.get('image_caption'))}")
    if block.get("diagram") and _s(_dict(block["diagram"]).get("caption")):
        parts.append(f"Schéma : {_s(block['diagram']['caption'])}")
    if block.get("chart"):
        parts.append(_chart_text(_dict(block["chart"])))
    return "\n".join(p for p in parts if p)


def _blocks_text(blocks: Any) -> list[str]:
    return [t for t in (block_text(b) for b in _items(blocks)) if t]


def _question_text(question: dict) -> str:
    options = [f"  - {_s(o)}" for o in _items(question.get("options"))]
    lines = [f"Question : {_s(question.get('question'))}", *options]
    if _s(question.get("explanation")):
        lines.append(f"  Explication : {_s(question.get('explanation'))}")
    return "\n".join(lines)


def _legacy_section_body(section: dict) -> list[str]:
    parts = [
        f"{label} : {_s(section.get(key))}"
        for key, label in (("pourquoi", "Pourquoi"), ("quoi", "Quoi"), ("comment", "Comment"))
        if _s(section.get(key))
    ]
    if section.get("worked_example"):
        parts.append(_worked_example_text(_dict(section["worked_example"])))
    parts.extend(_table_text(_dict(t)) for t in _items(section.get("tables")))
    return parts


def _section_text(index: int, section: dict) -> str:
    lines = [f"## Section {index} — {_s(section.get('title'))}"]
    if _s(section.get("challenge")):
        lines.append(f"Défi : {_s(section.get('challenge'))}")
    subsections = _items(section.get("subsections"))
    if subsections:
        for sub in subsections:
            sub = _dict(sub)
            if _s(sub.get("title")):
                lines.append(f"### {_s(sub.get('title'))}")
            lines.extend(_blocks_text(sub.get("blocks")))
    else:
        lines.extend(_legacy_section_body(section))
    key_points = [_s(p) for p in _items(section.get("key_points")) if _s(p)]
    if key_points:
        lines.append("Points clés :\n" + "\n".join(f"- {p}" for p in key_points))
    faded = _dict(section.get("faded_example"))
    if faded:
        steps = [*_items(faded.get("given_steps")), *_items(faded.get("hidden_steps"))]
        lines.append(_worked_example_text({**faded, "steps": steps}))
    lines.extend(_question_text(_dict(q)) for q in _items(section.get("check_questions")))
    return "\n".join(line for line in lines if line)


def _answer_text(answer: dict) -> list[str]:
    parts = [f"Réponse directe : {_s(answer.get('summary'))}" if _s(answer.get("summary")) else ""]
    parts.extend(_blocks_text(answer.get("blocks")))
    parts.extend(_legacy_section_body(answer))
    parts.extend(f"- {_s(p)}" for p in _items(answer.get("key_points")) if _s(p))
    return parts


_WORD_RE = re.compile(r"[a-z0-9]{3,}")
_STOPWORDS = frozenset(
    "les des une aux que qui quoi est sont pour par sur dans avec sans cette ces mais plus comme "
    "donc alors aussi tout tous elle elles ils nous vous leur leurs son ses mes tes the and "
    "est-ce quel quelle quels quelles peux peut faire fait etre avoir cela celui celle expliquer "
    "explique peut-tu moi pourquoi comment exemple".split()
)
_PREVIOUS_QUESTION_WEIGHT = 0.5
_TITLE_WEIGHT = 3


def _tokens(text: str) -> list[str]:
    """Mots significatifs (sans accents, minuscules, sans mots vides) pour le classement lexical."""
    folded = unicodedata.normalize("NFKD", text.lower())
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return [w for w in _WORD_RE.findall(folded) if w not in _STOPWORDS]


def _section_title_text(section: dict) -> str:
    titles = [_s(section.get("title"))] + [_s(_dict(sub).get("title")) for sub in _items(section.get("subsections"))]
    return " ".join(t for t in titles if t)


def rank_sections(sections: list[dict], queries: list[tuple[str, float]]) -> list[tuple[int, float]]:
    """Classe les sections (indices, score décroissant, score > 0) selon des requêtes pondérées.

    Score lexical local de type TF-IDF : aucun appel Gemini, donc aucun token consommé. Les mots du
    titre de la section et de ses sous-sections comptent `_TITLE_WEIGHT` fois plus.
    """
    docs: list[Counter] = []
    for section in sections:
        counts = Counter(_tokens(_section_text(0, section)))
        for token in _tokens(_section_title_text(section)):
            counts[token] += _TITLE_WEIGHT
        docs.append(counts)
    n = len(docs)
    scores = [0.0] * n
    for query, weight in queries:
        for token in set(_tokens(query)):
            df = sum(1 for d in docs if token in d)
            if not df:
                continue
            idf = math.log(1 + n / df)
            for i, d in enumerate(docs):
                if token in d:
                    scores[i] += weight * idf * (1 + math.log(d[token]))
    return sorted(((i, s) for i, s in enumerate(scores) if s > 0), key=lambda t: -t[1])


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - len(TRUNCATION_MARKER))].rstrip() + TRUNCATION_MARKER


def select_lesson_context(
    gemini_response: dict,
    message: str,
    *,
    previous_question: str = "",
    section_id: str | None = None,
    max_chars: int,
    top_sections: int = 2,
) -> str:
    """Contexte réduit : plan du cours + seulement les sections visées par la question.

    Sections retenues : `section_id` (si connu), puis les mieux classées pour `message` (et, avec un
    poids moindre, la question précédente pour les relances courtes). Sans correspondance : plan +
    introduction/synthèse tronquées. Sans sections (cours « réponse directe ») : cours borné.
    """
    course = _dict(gemini_response)
    sections = [_dict(s) for s in _items(course.get("sections"))]
    if not sections:
        return build_lesson_context(course, max_chars)

    title = _s(_dict(course.get("meta")).get("title"))
    outline = "\n".join(f"{i}. {_s(s.get('title'))}" for i, s in enumerate(sections, start=1))
    header = (f"# {title}\n" if title else "") + f"Plan du cours :\n{outline}"

    chosen: list[int] = []
    if section_id:
        chosen += [i for i, s in enumerate(sections) if str(s.get("id")) == str(section_id)][:1]
    queries = [(message, 1.0)] + ([(previous_question, _PREVIOUS_QUESTION_WEIGHT)] if previous_question else [])
    for i, _score in rank_sections(sections, queries):
        if len(chosen) >= max(1, top_sections):
            break
        if i not in chosen:
            chosen.append(i)

    budget = max(0, max_chars - len(header))
    if not chosen:
        extras = [_s(v) for v in _dict(course.get("introduction")).values()]
        if _s(course.get("summary")):
            extras.append(f"Synthèse : {_s(course.get('summary'))}")
        return header + "\n\n" + _truncate("\n\n".join(e for e in extras if e), budget)

    share = budget // len(chosen)
    parts = [_truncate(_section_text(i + 1, sections[i]), share) for i in sorted(chosen)]
    return header + "\n\n" + "\n\n".join(parts)


def build_lesson_context(gemini_response: dict, max_chars: int) -> str:
    """Rend le cours en texte (Markdown léger), borné à `max_chars` caractères."""
    course = _dict(gemini_response)
    title = _s(_dict(course.get("meta")).get("title"))
    parts: list[str] = [f"# {title}" if title else ""]
    intro = course.get("introduction")
    if isinstance(intro, dict):
        parts.extend(_s(v) for v in intro.values())
    parts.extend(_answer_text(_dict(course.get("answer"))))
    parts.extend(_section_text(i, _dict(s)) for i, s in enumerate(_items(course.get("sections")), start=1))
    pitfalls = [_pitfall_text(_dict(p)) for p in _items(course.get("common_pitfalls"))]
    if any(pitfalls):
        parts.append("## Pièges courants\n" + "\n\n".join(p for p in pitfalls if p))
    if _s(course.get("summary")):
        parts.append(f"## Synthèse\n{_s(course.get('summary'))}")
    next_steps = [_s(n) for n in _items(course.get("next_steps")) if _s(n)]
    if next_steps:
        parts.append("## Pour aller plus loin\n" + "\n".join(f"- {n}" for n in next_steps))

    text = "\n\n".join(p for p in parts if p)
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - len(TRUNCATION_MARKER))].rstrip() + TRUNCATION_MARKER
