from app.api.schemas import CourseSection
from app.schemas.course_generation import Section
from app.services.course_generator import _map_blocks, _map_sections_to_course_sections
from app.services.visual_validation import visual_issues


def _section(blocks: list[dict], type_: str = "development", title: str = "S") -> Section:
    return Section.model_validate(
        {"type": type_, "title": title, "blocks": [], "subsections": [{"title": "Quoi", "blocks": blocks}]}
    )


ALL_BLOCKS = [
    {"type": "text", "text": "Une phrase."},
    {"type": "definition", "text": "Terme : sens."},
    {"type": "list", "list_items": ["a", "b"], "list_ordered": True},
    {"type": "table", "table": {"caption": "c", "headers": ["h"], "rows": [["x"]]}},
    {"type": "formula", "formula": {"latex": "E=mc^2", "description": "énergie"}},
    {"type": "code", "code": "print(1)", "code_language": "python"},
    {"type": "worked_example", "worked_example": {"statement": "s", "steps": ["1", "2"], "result": "r"}},
    {"type": "callout", "text": "attention", "callout_variant": "warning"},
    {"type": "pitfall", "pitfall": {"description": "d", "why_it_happens": "w", "how_to_avoid": "h"}},
    {"type": "diagram", "diagram": {"kind": "flowchart", "caption": "flux", "mermaid": "flowchart TD; A-->B"}},
    {"type": "chart", "chart": {"kind": "bar", "caption": "ventes", "labels": ["a", "b"],
                                "series": [{"name": "n", "values": [1, 2]}]}},
]


def test_map_blocks_keeps_every_type_in_order():
    blocks = _section(ALL_BLOCKS).subsections[0].blocks

    mapped = _map_blocks(blocks)

    assert [b.type for b in mapped] == [b["type"] for b in ALL_BLOCKS]
    by_type = {b.type: b for b in mapped}
    assert by_type["diagram"].diagram.mermaid == "flowchart TD; A-->B"
    assert by_type["chart"].chart.series[0].values == [1, 2]
    assert by_type["formula"].formula.latex == "E=mc^2"
    assert by_type["worked_example"].worked_example.steps == ["1", "2"]
    assert by_type["callout"].callout_variant == "warning"


def test_oversized_diagram_is_dropped_and_chart_truncated():
    huge = {"type": "diagram", "diagram": {"kind": "flowchart", "mermaid": "A-->B;" * 2000}}
    wide = {"type": "chart", "chart": {"kind": "line", "labels": [str(i) for i in range(20)],
                                       "series": [{"name": "n", "values": [float(i) for i in range(20)]}]}}
    mapped = _map_blocks(_section([huge, wide]).subsections[0].blocks)

    assert [b.type for b in mapped] == ["chart"]
    assert len(mapped[0].chart.labels) == 12 and len(mapped[0].chart.series[0].values) == 12


def test_section_exposes_subsections_and_keeps_legacy_fields():
    api = _map_sections_to_course_sections([_section(ALL_BLOCKS)])[0]

    assert api.subsections[0].title == "Quoi"
    assert len(api.subsections[0].blocks) == len(ALL_BLOCKS)
    assert api.quoi  # champ déprécié toujours alimenté
    assert len(api.tables) == 1


def test_visual_only_section_is_not_dropped():
    diagram = {"type": "diagram", "diagram": {"kind": "cycle", "mermaid": "flowchart LR; A-->B; B-->A"}}
    sections = _map_sections_to_course_sections([_section([diagram])])

    assert len(sections) == 1 and sections[0].subsections[0].blocks[0].type == "diagram"


def test_legacy_section_without_subsections_still_valid():
    legacy = {
        "id": "0", "title": "T", "quoi": "q", "pourquoi": "p", "comment": "c",
        "worked_example": {"statement": "", "steps": [], "result": ""},
    }
    assert CourseSection.model_validate(legacy).subsections == []


def test_visual_issues():
    assert visual_issues(_section([{"type": "text", "text": "Une phrase."}])) == [
        "aucun bloc visuel (TABLE, LIST, DIAGRAM, CHART, FORMULA...)"
    ]
    long_text = {"type": "text", "text": "Un. Deux. Trois. Quatre."}
    visual = {"type": "list", "list_items": ["a"], "list_ordered": False}
    assert len(visual_issues(_section([long_text, visual]))) == 1
    assert visual_issues(_section([{"type": "text", "text": "Court."}, visual])) == []
    assert visual_issues(_section([long_text], type_="summary")) == []


def test_visual_issues_image_block_satisfies_visual_first_for_now():
    """Tant qu'aucun résolveur d'images réel n'est branché (Lot 1 : tout bloc IMAGE est retiré),
    IMAGE compte comme le visuel de la règle — sinon la réparation de complétude (voir
    `test_course_plan_completeness.py`) regénère la section via un appel Gemini payant qui ne peut
    de toute façon jamais aboutir à une image affichée. À inverser une fois un résolveur réel
    branché (Lot 2/3/5, voir visual_validation.py)."""
    image_only = {"type": "image", "image_source": "web", "image_query": "chat noir"}
    assert visual_issues(_section([image_only])) == []
