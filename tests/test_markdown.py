import pytest

from app.api.schemas import ApiPlannedSection
from app.core.markdown import markdown_to_plain
from app.services.course_plan_generator import _incomplete_section, _is_covered, _norm


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("**Gras** et *italique*", "Gras et italique"),
        ("~~barré~~ et `code`", "barré et code"),
        ("_souligné_ mais snake_case et x_1", "souligné mais snake_case et x_1"),
        ("- premier\n- second\n1. troisième", "premier\nsecond\ntroisième"),
        ("```python\nprint(1)\n```", "print(1)"),
        ("[lien](https://exemple.fr) et ![img](a.png)", "lien et img"),
        ("Formule $a_b + c_d$ intacte", "Formule $a_b + c_d$ intacte"),
        (r"Échappé : 2 \* 3 \_x", "Échappé : 2 * 3 _x"),
        ("", ""),
        ("Texte brut", "Texte brut"),
    ],
)
def test_markdown_to_plain(source, expected):
    assert markdown_to_plain(source) == expected


def test_markdown_to_plain_is_idempotent_on_plain_text():
    plain = markdown_to_plain("**Loi** d'_Ohm_ : `U = R * I`")
    assert markdown_to_plain(plain) == plain


def test_norm_ignores_markdown_markers():
    assert _norm("**Loi d'Ohm** et _tension_") == _norm("loi d'ohm et tension")
    assert _is_covered("**Loi d'Ohm**", {_norm("La loi d'Ohm en courant continu")})


def test_incomplete_section_shows_plan_as_plain_text():
    planned = ApiPlannedSection(
        type="development", title="Ohm", objective="Comprendre **U = RI**", subtopics=["*tension*", "`R`"], order=1
    )
    section = _incomplete_section(planned)
    assert section.subsections[0].blocks[0].text == "Comprendre U = RI"
    assert section.subsections[1].blocks[0].text == "Points prévus : tension ; R"
