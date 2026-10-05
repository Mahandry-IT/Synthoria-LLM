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


# --- Sélection des sections pertinentes (chat : on n'envoie plus le cours entier) ---

SELECT_COURSE = {
    "meta": {"title": "Électricité"},
    "introduction": {"why": "Introduction générale."},
    "summary": "Synthèse du cours.",
    "sections": [
        {"id": "s1", "title": "Le transformateur", "subsections": [
            {"title": "Rapport de transformation", "blocks": [{"type": "text", "text": "Le rapport des spires fixe la tension."}]}]},
        {"id": "s2", "title": "Les condensateurs", "subsections": [
            {"title": "Capacité", "blocks": [{"type": "text", "text": "Un condensateur stocke une charge électrique."}]}]},
        {"id": "s3", "title": "Les diodes", "subsections": [
            {"title": "Redressement", "blocks": [{"type": "text", "text": "La diode laisse passer le courant dans un sens."}]}]},
    ],
}


def test_select_keeps_outline_and_only_the_matching_section():
    from app.services.lesson_context import select_lesson_context

    context = select_lesson_context(SELECT_COURSE, "Comment fonctionne un condensateur ?", max_chars=5000, top_sections=1)

    assert "Plan du cours" in context and "1. Le transformateur" in context and "3. Les diodes" in context
    assert "stocke une charge" in context
    assert "fixe la tension" not in context and "laisse passer" not in context


def test_select_is_accent_insensitive_and_weights_titles():
    from app.services.lesson_context import rank_sections

    ranking = rank_sections(SELECT_COURSE["sections"], [("redressement diode", 1.0)])

    assert ranking and ranking[0][0] == 2


def test_select_section_id_is_prioritary_and_unknown_id_is_ignored():
    from app.services.lesson_context import select_lesson_context

    forced = select_lesson_context(
        SELECT_COURSE, "condensateur", section_id="s3", max_chars=5000, top_sections=2
    )
    assert "laisse passer" in forced and "stocke une charge" in forced

    unknown = select_lesson_context(SELECT_COURSE, "condensateur", section_id="zzz", max_chars=5000, top_sections=1)
    assert "stocke une charge" in unknown and "laisse passer" not in unknown


def test_select_short_followup_uses_previous_question():
    from app.services.lesson_context import select_lesson_context

    context = select_lesson_context(
        SELECT_COURSE, "Et pour la suite ?", previous_question="le transformateur", max_chars=5000, top_sections=1
    )

    assert "fixe la tension" in context


def test_select_without_match_falls_back_to_outline_intro_and_summary():
    from app.services.lesson_context import select_lesson_context

    context = select_lesson_context(SELECT_COURSE, "blabla xyz", max_chars=5000)

    assert "Plan du cours" in context and "Introduction générale." in context and "Synthèse du cours." in context
    assert "fixe la tension" not in context


def test_select_respects_the_character_budget():
    from app.services.lesson_context import select_lesson_context

    big = {**SELECT_COURSE, "sections": [
        {**SELECT_COURSE["sections"][0], "subsections": [
            {"title": "Rapport", "blocks": [{"type": "text", "text": "transformateur " * 2000}]}]},
        *SELECT_COURSE["sections"][1:],
    ]}

    assert len(select_lesson_context(big, "transformateur", max_chars=1500)) <= 1500 + 60
