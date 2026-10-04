from app.services.lesson_context import TRUNCATION_MARKER, block_text, build_lesson_context

COURSE = {
    "meta": {"title": "Le transformateur"},
    "introduction": {"contexte": "Un transformateur adapte la tension."},
    "sections": [
        {
            "id": "0",
            "title": "Rapport de transformation",
            "challenge": "Que se passe-t-il si on double les spires ?",
            "subsections": [
                {
                    "title": "Quoi",
                    "blocks": [
                        {"type": "text", "text": "Le rapport m vaut N2/N1."},
                        {"type": "list", "list_items": ["primaire", "secondaire"], "list_ordered": True},
                        {"type": "table", "table": {"caption": "Valeurs", "headers": ["N1", "N2"], "rows": [["100", "200"]]}},
                        {"type": "formula", "formula": {"latex": "m = N_2/N_1", "description": "rapport"}},
                        {"type": "code", "code": "print(m)", "code_language": "python"},
                        {"type": "worked_example", "worked_example": {"statement": "N1=100", "steps": ["m=2"], "result": "U2 double"}},
                        {"type": "pitfall", "pitfall": {"description": "Inverser N1/N2", "why_it_happens": "confusion", "how_to_avoid": "vérifier"}},
                    ],
                }
            ],
            "check_questions": [{"question": "Que vaut m ?", "options": ["1", "2"], "explanation": "N2/N1"}],
        },
        {
            "id": "1",
            "title": "Ancienne section",
            "quoi": "Définition héritée.",
            "worked_example": {"statement": "Énoncé", "steps": [{"id": "1", "content": "Étape héritée"}], "result": "OK"},
        },
    ],
    "common_pitfalls": [{"description": "Oublier les pertes", "why_it_happens": "", "how_to_avoid": ""}],
    "summary": "Le rapport fixe la tension.",
    "next_steps": ["Pertes fer"],
}


def test_renders_every_kind_of_content():
    text = build_lesson_context(COURSE, 100_000)

    for expected in (
        "# Le transformateur", "Un transformateur adapte la tension.", "Section 1 — Rapport de transformation",
        "Défi : Que se passe-t-il", "Le rapport m vaut N2/N1.", "1. primaire", "N1 | N2", "100 | 200",
        "$$m = N_2/N_1$$ (rapport)", "```python\nprint(m)\n```", "Exemple : N1=100", "Résultat : U2 double",
        "Piège : Inverser N1/N2", "Question : Que vaut m ?", "Explication : N2/N1",
        "Quoi : Définition héritée.", "Étape héritée", "Oublier les pertes", "Le rapport fixe la tension.",
        "- Pertes fer",
    ):
        assert expected in text


def test_is_bounded_with_marker():
    text = build_lesson_context(COURSE, 200)

    assert len(text) <= 200 and text.endswith(TRUNCATION_MARKER)


def test_tolerates_missing_or_malformed_fields():
    assert build_lesson_context({}, 1000) == ""
    assert build_lesson_context({"sections": [None, "x", {"subsections": [{"blocks": [None]}]}]}, 1000)
    assert block_text({"type": "chart", "chart": {"caption": "Ventes", "labels": ["a"], "series": [{"name": "s", "values": [1]}]}}) == (
        "Graphique : Ventes\ns : a=1"
    )
