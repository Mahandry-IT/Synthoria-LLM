"""La consigne de formatage du code doit aussi viser l'énoncé des exemples (régression : du C non
balisé dans `statement` était découpé en liste à puces sur les « ; » par le frontend)."""

import pytest

from app.schemas.course_generation import FadedExample, WorkedExample


@pytest.mark.parametrize("model", [WorkedExample, FadedExample])
def test_statement_description_forbids_bare_code(model):
    description = model.model_fields["statement"].description

    assert "never bare" in description and "fenced" in description and "backticks" in description
