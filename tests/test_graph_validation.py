import pytest
from pydantic import ValidationError

from app.api.schemas import GRAPH_EDGES_MAX, GRAPH_NODES_MAX, GRAPH_STEPS_MAX, ApiGraph


def _nodes(*ids: str) -> list[dict]:
    return [{"id": i, "label": i.upper()} for i in ids]


def _edge(edge_id: str, source: str, target: str) -> dict:
    return {"id": edge_id, "source": source, "target": target}


def _graph(**overrides) -> dict:
    base = {
        "layout": "graph", "directed": False,
        "nodes": _nodes("a", "b", "c"),
        "edges": [_edge("e1", "a", "b"), _edge("e2", "b", "c")],
    }
    return {**base, **overrides}


def test_valid_graph_with_steps_and_cycle_is_accepted():
    graph = ApiGraph.model_validate(_graph(
        edges=[_edge("e1", "a", "b"), _edge("e2", "b", "c"), _edge("e3", "c", "a"), _edge("e4", "a", "a")],
        steps=[{"caption": "Départ.", "highlight_nodes": ["a"]}, {"caption": "Fin.", "highlight_edges": ["e3"]}],
    ))

    assert [s.highlight_nodes for s in graph.steps] == [["a"], []]


@pytest.mark.parametrize("overrides", [
    {"nodes": _nodes("a", "a", "b")},  # identifiant de nœud en double
    {"edges": [_edge("e1", "a", "b"), _edge("e1", "b", "c")]},  # identifiant d'arête en double
    {"edges": [_edge("e1", "a", "z")]},  # arête vers un nœud inconnu
    {"steps": [{"caption": "x", "highlight_nodes": ["z"]}]},  # étape : nœud inconnu
    {"steps": [{"caption": "x", "highlight_edges": ["e9"]}]},  # étape : arête inconnue
    {"steps": [{"caption": "", "highlight_nodes": ["a"]}]},  # étape sans texte
    {"nodes": _nodes("a")},  # un seul nœud
    {"edges": []},  # aucune arête
    {"layout": "mindmap"},  # disposition inconnue
])
def test_inconsistent_graph_is_rejected(overrides):
    with pytest.raises(ValidationError):
        ApiGraph.model_validate(_graph(**overrides))


def test_graph_beyond_render_limits_is_rejected():
    many = [str(i) for i in range(GRAPH_NODES_MAX + 1)]
    too_many_edges = [_edge(f"e{i}", "a", "b") for i in range(GRAPH_EDGES_MAX + 1)]
    too_many_steps = [{"caption": "x"}] * (GRAPH_STEPS_MAX + 1)

    for overrides in ({"nodes": _nodes(*many)}, {"edges": too_many_edges}, {"steps": too_many_steps}):
        with pytest.raises(ValidationError):
            ApiGraph.model_validate(_graph(**overrides))


def test_valid_tree_is_accepted():
    tree = _graph(layout="tree", directed=True, nodes=_nodes("r", "a", "b", "c"),
                  edges=[_edge("e1", "r", "a"), _edge("e2", "r", "b"), _edge("e3", "a", "c")])

    assert ApiGraph.model_validate(tree).layout == "tree"


@pytest.mark.parametrize("overrides", [
    {"directed": False},  # arbre non orienté
    {"edges": [_edge("e1", "r", "a"), _edge("e2", "r", "b"), _edge("e3", "a", "b")]},  # deux parents
    {"edges": [_edge("e1", "r", "a")]},  # deux racines (r et b)
    {"edges": [_edge("e1", "a", "b"), _edge("e2", "b", "a")]},  # cycle détaché de la racine r
    {"edges": [_edge("e1", "r", "a"), _edge("e2", "a", "b"), _edge("e3", "b", "r")]},  # cycle, aucune racine
])
def test_invalid_tree_is_rejected(overrides):
    tree = {"layout": "tree", "directed": True, "nodes": _nodes("r", "a", "b"),
            "edges": [_edge("e1", "r", "a"), _edge("e2", "r", "b")]}

    with pytest.raises(ValidationError):
        ApiGraph.model_validate({**tree, **overrides})


# ─── Schémas envoyés à Gemini ──────────────────────────────────


def test_whole_course_gemini_schema_does_not_offer_the_graph_block():
    """Avec GRAPH, le schéma « cours entier » est rejeté par Gemini (400 INVALID_ARGUMENT, constaté
    en réel) : il ne doit le proposer sous aucune forme, sinon toute génération directe échoue et
    un cours généré depuis un plan perd introduction, pièges, résumé et quiz."""
    import json

    from app.schemas.course_generation import COURSE_GENERATION_GEMINI_SCHEMA, CourseGenerationSchema, SectionsBatchSchema

    defs = COURSE_GENERATION_GEMINI_SCHEMA["$defs"]
    assert "graph" not in defs["ContentBlock"]["properties"]
    assert "graph" not in defs["BlockType"]["enum"]
    assert not [name for name in defs if name.startswith("Graph")]
    assert "#/$defs/Graph" not in json.dumps(COURSE_GENERATION_GEMINI_SCHEMA)  # aucune référence orpheline
    # Seul GRAPH est retiré : le reste du schéma est celui de CourseGenerationSchema.
    full = CourseGenerationSchema.model_json_schema()
    assert COURSE_GENERATION_GEMINI_SCHEMA["properties"] == full["properties"]
    assert set(full["$defs"]) - set(defs) == {"GraphData", "GraphNode", "GraphEdge", "GraphStep", "GraphLayout"}
    # Les sections de développement, elles, continuent de le proposer.
    assert "GraphData" in SectionsBatchSchema.model_json_schema()["$defs"]


def test_strip_graph_block_never_mutates_the_schema_and_ignores_schemas_without_it():
    from app.schemas.course_generation import SectionsBatchSchema, VideoRankingSchema, strip_graph_block

    original = SectionsBatchSchema.model_json_schema()
    stripped = strip_graph_block(original)

    assert "GraphData" in original["$defs"] and "graph" in original["$defs"]["ContentBlock"]["properties"]
    assert "GraphData" not in stripped["$defs"]
    unrelated = VideoRankingSchema.model_json_schema()
    assert strip_graph_block(unrelated) is unrelated
    assert strip_graph_block({}) == {}
