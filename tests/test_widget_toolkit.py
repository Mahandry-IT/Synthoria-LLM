"""Boîte à outils `window.SW` : cohérence du fichier de signatures, du prompt et du filtre."""

import json
from pathlib import Path

from app.api.schemas import widget_html_problem
from app.services import course_widgets
from tests.test_course_widgets import _response

_TOOLKIT = json.loads((Path(course_widgets.__file__).with_name("widget_toolkit.json")).read_text(encoding="utf-8"))


def _prompt() -> str:
    response = _response()
    return course_widgets._prompt(response, response.sections[1], 20000, None)


def test_prompt_cites_every_toolkit_function() -> None:
    prompt = _prompt()
    assert _TOOLKIT["functions"]
    for fn in _TOOLKIT["functions"]:
        assert fn["signature"] in prompt
        assert f"SW.{fn['name']}(" in prompt


def test_prompt_rules() -> None:
    prompt = _prompt()
    assert "sauf `window.SW`" in prompt
    assert "SW.slider" in prompt and "SW.stepper" in prompt
    assert "PAS du SVG écrit à la main" in prompt
    assert _TOOLKIT["example_widget"] in prompt


def test_example_widget_passes_filter() -> None:
    assert widget_html_problem(_TOOLKIT["example_widget"]) is None


def test_every_function_example_passes_filter() -> None:
    for fn in _TOOLKIT["functions"]:
        assert widget_html_problem(f"<script>{fn['example']}</script>") is None, fn["name"]


def test_filter_still_rejects_forbidden_patterns_with_sw() -> None:
    assert widget_html_problem("<script>SW.line(c, []); fetch('x')</script>") is not None
    assert widget_html_problem("<script>SW.svg(); window.parent.x</script>") is not None
    assert widget_html_problem("<script src='a.js'></script>") is not None
