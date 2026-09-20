import pytest

from app.api.schemas import CourseGenerationResponse
from app.services.podcast.course_serializer import has_usable_content, serialize_course
from app.services.podcast.tts_normalizer import latex_to_words, normalize_for_tts, split_for_tts


# ─── Normalisation TTS ───────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Cela coûte 50 €", "Cela coûte 50 euros"),
        ("Un taux de 85 %", "Un taux de 85 pour cent"),
        ("Le réseau 10.0.0.0/24", "Le réseau 10 point 0 point 0 point 0 slash 24"),
        ("Le serveur 192.168.1.10", "Le serveur 192 point 168 point 1 point 10"),
        ("La valeur 0.145", "La valeur 0,145"),
        ("Consommation de 12 kWh", "Consommation de 12 kilowattheures"),
        ("Le PUE vaut 1,5", "Le P U E vaut 1,5"),
        ("Émissions en CO2e", "Émissions en C O deux équivalent"),
        ("Pour A et/ou B", "Pour A et ou B"),
        ("Soit 3/4 des cas", "Soit 3 sur 4 des cas"),
    ],
)
def test_normalize_spoken_forms(raw, expected):
    assert normalize_for_tts(raw) == expected


def test_normalize_removes_markdown_and_code_and_control_chars():
    raw = "## Titre\n**gras** et `int x = 0;`\x07\n```python\nprint(1)\n```\n- puce [lien](http://x)"
    out = normalize_for_tts(raw)
    assert "#" not in out and "**" not in out and "`" not in out and "http" not in out
    assert "print" not in out
    assert "extrait de code" in out
    assert "int x égale 0;" in out or "int x égale 0" in out
    assert "\x07" not in out


def test_normalize_latex_never_leaks_backslashes():
    out = normalize_for_tts(r"On pose $y = \beta_0 + \beta_1 x + \epsilon$ et $\frac{a}{b}$ puis $x^2$.")
    assert "\\" not in out and "$" not in out and "{" not in out
    assert "bêta" in out and "epsilon" in out and "a sur b" in out and "au carré" in out


def test_latex_to_words_handles_unknown_commands():
    assert "\\" not in latex_to_words(r"\foo{x}")


def test_normalize_unicode_nfkc():
    assert normalize_for_tts("ﬁn du cours") == "fin du cours"


def test_normalize_custom_acronyms_can_be_disabled():
    assert normalize_for_tts("Le PUE", acronyms={}) == "Le PUE"


def test_split_keeps_chunks_under_limit_and_loses_no_words():
    text = ("Voici une phrase assez longue, avec des virgules; et des points-virgules, pour tester. " * 12).strip()
    chunks = split_for_tts(text, max_chars=120)
    assert all(len(c) <= 120 for c in chunks)
    assert " ".join(chunks).split() == text.split()


def test_split_handles_oversized_word():
    chunks = split_for_tts("a" * 500, max_chars=100)
    assert all(len(c) <= 100 for c in chunks)
    assert "".join(chunks) == "a" * 500


def test_split_short_text_single_chunk():
    assert split_for_tts("Bonjour. Ça va ?", max_chars=400) == ["Bonjour. Ça va ?"]


# ─── Sérialisation du cours ──────────────────────────────────


def _course(**overrides) -> CourseGenerationResponse:
    base = {
        "mode": "question_only",
        "format": "full_course",
        "meta": {"title": "Green IT", "subject": "S", "language": "fr", "generated_at": "2026-09-20T10:00:00Z"},
        "sources": [],
        "introduction": {"quoi": "Le Green IT réduit l'impact du numérique."},
        "sections": [
            {
                "id": "0", "title": "ACV",
                "quoi": "L'analyse de cycle de vie mesure $CO_2$.",
                "pourquoi": "Pour comparer.",
                "comment": "On applique $$E = m c^2$$ (énergie et masse) puis ```python\nprint(1)\n```.",
                "worked_example": {
                    "statement": "Un serveur de 200 W.",
                    "steps": [{"id": "1", "content": "Multiplier par `24` heures."}],
                    "result": "4,8 kWh par jour.",
                },
                "key_points": ["Mesurer avant d'agir"],
            },
        ],
        "common_pitfalls": [
            {"description": "Ignorer la fabrication.", "why_it_happens": "Biais.", "how_to_avoid": "Inclure l'ACV."}
        ],
        "quiz": [
            {"question": "Q ?", "options": ["a", "b"], "correct_option_indices": [0]},
        ],
        "summary": "Retenez l'essentiel.",
        "next_steps": ["Écoconception", "Sobriété"],
    }
    base.update(overrides)
    return CourseGenerationResponse.model_validate(base)


def test_serialize_orders_sections_and_excludes_quiz():
    sections = serialize_course(_course())
    assert [(s.kind, s.title) for s in sections] == [
        ("intro", "Introduction"),
        ("development", "ACV"),
        ("pitfalls", "Pièges fréquents"),
        ("summary", "Synthèse"),
        ("next_steps", "Pour aller plus loin"),
    ]
    assert [s.index for s in sections] == [0, 1, 2, 3, 4]
    assert all("Q ?" not in s.text for s in sections)


def test_serialize_never_exposes_raw_latex_or_code():
    text = serialize_course(_course())[1].text
    assert "$" not in text and "\\" not in text and "print(1)" not in text and "`" not in text
    assert "énergie et masse" in text  # la description de la formule remplace le LaTeX
    assert "extrait de code" in text
    assert "Étape 1 : Multiplier par 24 heures." in text
    assert "Points clés : Mesurer avant d'agir" in text


def test_serialize_accepts_persisted_dict():
    sections = serialize_course(_course().model_dump())
    assert len(sections) == 5


def test_serialize_focused_answer_without_sections():
    course = _course(
        format="focused_answer", introduction=None, sections=None, common_pitfalls=None,
        answer={
            "quoi": "Définition.", "pourquoi": "Raison.", "comment": "Méthode.",
            "worked_example": {"statement": "s", "steps": [], "result": "r"}, "key_points": [],
        },
    )
    sections = serialize_course(course)
    assert sections[0].kind == "development" and sections[0].title == "Green IT"
    assert has_usable_content(sections)


def test_serialize_skips_empty_sections_and_reports_no_usable_content():
    course = _course(
        introduction=None, sections=None, common_pitfalls=None, quiz=None, summary="   ", next_steps=[],
        format="focused_answer",
        answer={
            "quoi": "", "pourquoi": "", "comment": "",
            "worked_example": {"statement": "", "steps": [], "result": ""}, "key_points": [],
        },
    )
    sections = serialize_course(course)
    assert sections == []
    assert not has_usable_content(sections)
