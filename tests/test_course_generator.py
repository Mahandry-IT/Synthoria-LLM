from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.config import Settings
from app.core.exceptions import (
    GeminiInvalidResponseError,
    GeminiUnavailableError,
)
from app.schemas.course_generation import QuizDifficulty, QuizQuestion
from app.services.course_generator import (
    _is_quiz_difficulty_error,
    _map_sections_to_course_sections,
    _rebalance_quiz_difficulty,
    _validate_and_map,
    _validate_and_map_with_retry,
    compute_quiz_points,
    generate_course_from_question,
)

VALID_STRUCTURED_ANSWER_FILE = {
    "mode": "file_question",
    "format": "focused_answer",
    "meta": {
        "title": "Transformateur",
        "subject": "Electrotechnique",
        "language": "fr",
        "generated_at": "2026-08-19T10:00:00Z",
    },
    "sources": [{"type": "file_chunk", "label": "doc.pdf", "reference": "doc_1_chunk_0"}],
    "sections": [
        {
            "type": "development",
            "title": "Le transformateur",
            "blocks": [],
            "subsections": [
                {"title": "Quoi", "blocks": [{"type": "text", "text": "definition"}]},
                {"title": "Pourquoi", "blocks": [{"type": "text", "text": "raison"}]},
                {"title": "Comment", "blocks": [{"type": "text", "text": "mecanisme"}]},
            ],
        },
    ],
    "quiz": [],
    "confidence": "high",
    "unconfirmed_points": [],
}

VALID_STRUCTURED_ANSWER_QUESTION_ONLY = {
    "mode": "question_only",
    "format": "focused_answer",
    "meta": {
        "title": "Transformateur",
        "subject": "Electrotechnique",
        "language": "fr",
        "generated_at": "2026-08-19T10:00:00Z",
    },
    "sources": [{"type": "file_chunk", "label": "doc.pdf", "reference": "doc_1_chunk_0"}],
    "sections": [
        {
            "type": "development",
            "title": "Le transformateur",
            "blocks": [],
            "subsections": [
                {"title": "Quoi", "blocks": [{"type": "text", "text": "définition"}]},
                {"title": "Pourquoi", "blocks": [{"type": "text", "text": "raison"}]},
                {"title": "Comment", "blocks": [{"type": "text", "text": "mécanisme"}]},
            ],
        },
    ],
    "quiz": [],
    "confidence": "high",
    "unconfirmed_points": [],
}


@pytest.fixture
def settings() -> Settings:
    return Settings(gemini_api_key="fake-key", gemini_use_search_grounding=True)


@pytest.fixture
def settings_no_grounding() -> Settings:
    return Settings(gemini_api_key="fake-key", gemini_use_search_grounding=False)


@pytest.fixture
def vector_store():
    store = AsyncMock()
    store.search.return_value = [
        {"content": "extrait pertinent", "metadata": {"filename": "doc.pdf", "page": 2}, "distance": 0.1}
    ]
    return store


@pytest.fixture
def gemini_client():
    client = AsyncMock()
    client.search_grounded.return_value = ("réponse brute", [{"type": "web", "label": "W", "reference": "https://w"}])
    client.format_structured.return_value = VALID_STRUCTURED_ANSWER_FILE.copy()
    return client


@pytest.mark.asyncio
async def test_generate_course_file_question_happy_path(settings, vector_store, gemini_client):
    result = await generate_course_from_question(
        question="Comment fonctionne un transformateur ?",
        vector_store=vector_store,
        gemini_client=gemini_client,
        settings=settings,
        mode="file_question",
    )

    assert result.mode == "file_question"
    assert result.format == "focused_answer"
    assert result.sources[0].type == "file"
    assert result.sections is not None
    assert len(result.sections) > 0
    vector_store.search.assert_awaited_once()
    gemini_client.search_grounded.assert_awaited_once()
    gemini_client.format_structured.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_course_filters_by_filename(settings, gemini_client):
    """Le filtre filename doit être transmis à vector_store.search (pré-filtrage
    côté store), et non ré-appliqué après coup par l'orchestrateur."""
    vector_store = AsyncMock()
    vector_store.search.return_value = [
        {"content": "a", "metadata": {"filename": "doc1.pdf", "page": 1}, "distance": 0.1},
    ]
    vector_store.count_pages = MagicMock(return_value=0)  # pas de controle de couverture pour ce test

    await generate_course_from_question(
        question="question",
        vector_store=vector_store,
        gemini_client=gemini_client,
        settings=settings,
        filename="doc1.pdf",
    )

    vector_store.search.assert_awaited_once()
    search_kwargs = vector_store.search.call_args.kwargs
    assert search_kwargs["filename_filter"] == "doc1.pdf"
    assert search_kwargs["top_k"] == settings.course_top_k_default
    prompt_used = gemini_client.search_grounded.call_args.kwargs["prompt"]
    assert "doc1.pdf" in prompt_used


@pytest.mark.asyncio
async def test_generate_course_question_only_skips_retrieval(settings, gemini_client):
    vector_store = AsyncMock()

    await generate_course_from_question(
        question="question",
        vector_store=vector_store,
        gemini_client=gemini_client,
        settings=settings,
        mode="question_only",
    )

    vector_store.search.assert_not_awaited()


@pytest.mark.asyncio
async def test_generate_course_single_call_no_grounding(settings_no_grounding, vector_store, gemini_client):
    """Mode sans search grounding : 1 seul appel (format_structured), pas de search_grounded."""
    result = await generate_course_from_question(
        question="Question simple",
        vector_store=vector_store,
        gemini_client=gemini_client,
        settings=settings_no_grounding,
    )

    assert result.mode == "file_question"
    gemini_client.search_grounded.assert_not_awaited()
    gemini_client.format_structured.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_course_invalid_structured_response_raises(settings, vector_store):
    gemini_client = AsyncMock()
    gemini_client.search_grounded.return_value = ("réponse brute", [])
    gemini_client.format_structured.return_value = {"mode": "file_question"}  # incomplet, invalide

    with pytest.raises(GeminiInvalidResponseError):
        await generate_course_from_question(
            question="question",
            vector_store=vector_store,
            gemini_client=gemini_client,
            settings=settings,
        )


# --- Mode 3 : question_only (recherche web, pas de RAG) ---


@pytest.fixture
def gemini_client_question_only():
    client = AsyncMock()
    client.search_grounded.return_value = (
        "reponse brute web",
        [{"type": "web", "label": "W", "reference": "https://w"}],
    )
    client.format_structured.return_value = VALID_STRUCTURED_ANSWER_QUESTION_ONLY.copy()
    return client


@pytest.mark.asyncio
async def test_generate_course_question_only_with_grounding(
    settings, gemini_client_question_only
):
    """question_only + search grounding : search_grounded appele, vector_store non."""
    vector_store = AsyncMock()

    result = await generate_course_from_question(
        question="Qu'est-ce que l'IA ?",
        vector_store=vector_store,
        gemini_client=gemini_client_question_only,
        settings=settings,
        mode="question_only",
    )

    assert result.mode == "question_only"
    vector_store.search.assert_not_awaited()
    gemini_client_question_only.search_grounded.assert_awaited_once()
    gemini_client_question_only.format_structured.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_course_question_only_no_grounding(
    settings_no_grounding, gemini_client_question_only
):
    """question_only sans search grounding : 1 seul appel format_structured."""
    vector_store = AsyncMock()

    result = await generate_course_from_question(
        question="Question simple",
        vector_store=vector_store,
        gemini_client=gemini_client_question_only,
        settings=settings_no_grounding,
        mode="question_only",
    )

    assert result.mode == "question_only"
    vector_store.search.assert_not_awaited()
    gemini_client_question_only.search_grounded.assert_not_awaited()
    gemini_client_question_only.format_structured.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_course_question_only_prompt_no_rag_context(
    settings, gemini_client_question_only
):
    """Le prompt ne doit pas mentionner de contexte RAG pour question_only."""
    vector_store = AsyncMock()

    await generate_course_from_question(
        question="Explique la regression",
        vector_store=vector_store,
        gemini_client=gemini_client_question_only,
        settings=settings,
        mode="question_only",
    )

    prompt_used = gemini_client_question_only.search_grounded.call_args.kwargs["prompt"]
    assert "Contexte extrait des documents" not in prompt_used
    assert "Regression" in prompt_used or "regression" in prompt_used.lower()


# --- Multi-file search ---


@pytest.mark.asyncio
async def test_generate_course_multi_file_separate_searches(settings):
    """Avec filename=list, vector_store.search est appele une fois par fichier."""
    vector_store = AsyncMock()
    # Chaque appel retourne des chunks differents par fichier
    vector_store.search.side_effect = [
        [{"content": "chunk gradient", "metadata": {"filename": "gradient.pdf", "page": 1}, "distance": 0.1}],
        [{"content": "chunk regression", "metadata": {"filename": "regression.pdf", "page": 1}, "distance": 0.15}],
    ]
    vector_store.count_pages = MagicMock(return_value=0)  # pas de controle de couverture pour ce test
    gemini_client = AsyncMock()
    gemini_client.search_grounded.return_value = ("reponse", [])
    gemini_client.format_structured.return_value = VALID_STRUCTURED_ANSWER_FILE.copy()

    await generate_course_from_question(
        question="Compare les deux cours",
        vector_store=vector_store,
        gemini_client=gemini_client,
        settings=settings,
        filename=["gradient.pdf", "regression.pdf"],
    )

    # 2 appels search (un par fichier)
    assert vector_store.search.await_count == 2
    # Chaque appel doit filtrer sur SON fichier (pas de recherche globale
    # partagée qui écraserait les fichiers non dominants) : régression du
    # bug où seul le premier fichier finissait dans le contexte.
    called_filters = [c.kwargs["filename_filter"] for c in vector_store.search.await_args_list]
    assert set(called_filters) == {"gradient.pdf", "regression.pdf"}
    # Les deux chunks sont dans le contexte
    prompt_used = gemini_client.search_grounded.call_args.kwargs["prompt"]
    assert "chunk gradient" in prompt_used
    assert "chunk regression" in prompt_used


@pytest.mark.asyncio
async def test_generate_course_multi_file_real_store_both_files_represented(settings, gemini_client):
    """Régression bout-en-bout : avec un vrai NumpyVectorStore (pas de mock),
    deux fichiers ingérés doivent tous les deux apparaître dans le contexte,
    même si l'un des deux est globalement plus proche de la question pour
    TOUS ses chunks (cas qui reproduit le bug rapporté)."""
    from unittest.mock import AsyncMock as _AsyncMock

    from app.services.vector_store import NumpyVectorStore

    gemini_client.reformulate_query.return_value = "requete de recherche"

    ollama_client = _AsyncMock()
    # La requête reformulée s'aligne parfaitement avec les chunks de
    # "dominant.pdf" ; "minoritaire.pdf" a une similarité plus faible mais
    # doit quand même être représenté dans le contexte grâce au filtrage
    # par fichier AVANT le classement top_k (le bug reproduit ici serait
    # de ne récupérer que dominant.pdf).
    ollama_client.embed.side_effect = lambda text: (
        [1.0, 0.0] if text == "requete de recherche" else [0.0, 1.0]
    )

    store = NumpyVectorStore(settings=settings, ollama_client=ollama_client)
    store._documents = [
        {
            "id": "1",
            "content": "extrait dominant 1",
            "metadata": {"filename": "dominant.pdf", "page": 1},
            "embedding": [1.0, 0.0],
        },
        {
            "id": "2",
            "content": "extrait dominant 2",
            "metadata": {"filename": "dominant.pdf", "page": 2},
            "embedding": [0.99, 0.05],
        },
        {
            "id": "3",
            "content": "extrait minoritaire",
            "metadata": {"filename": "minoritaire.pdf", "page": 1},
            "embedding": [0.0, 1.0],
        },
    ]

    from app.services.course_generator import generate_course_from_question

    await generate_course_from_question(
        question="Donne moi le cours complet",
        vector_store=store,
        gemini_client=gemini_client,
        settings=settings,
        filename=["dominant.pdf", "minoritaire.pdf"],
    )

    prompt_used = gemini_client.search_grounded.call_args.kwargs["prompt"]
    assert "extrait minoritaire" in prompt_used
    assert "dominant" in prompt_used


# --- full_document mode ---


@pytest.mark.asyncio
async def test_generate_course_full_document_bypasses_search(settings, gemini_client):
    """full_document=True doit appeler get_all_chunks au lieu de search."""
    vector_store = AsyncMock()
    vector_store.get_all_chunks = MagicMock(
        return_value=[
            {"content": "page 1", "metadata": {"filename": "doc.pdf", "page": 1}, "distance": 0.0},
            {"content": "page 2", "metadata": {"filename": "doc.pdf", "page": 2}, "distance": 0.0},
        ]
    )
    vector_store.count_pages = MagicMock(return_value=0)  # isole le test du contrôle de couverture

    await generate_course_from_question(
        question="question",
        vector_store=vector_store,
        gemini_client=gemini_client,
        settings=settings,
        filename="doc.pdf",
        full_document=True,
    )

    vector_store.get_all_chunks.assert_called_once_with("doc.pdf")
    vector_store.search.assert_not_awaited()


@pytest.mark.asyncio
async def test_generate_course_full_document_false_keeps_existing_behavior(settings, vector_store, gemini_client):
    """full_document=False (défaut) : comportement inchangé, search() est utilisé."""
    vector_store.count_pages = MagicMock(return_value=0)

    result = await generate_course_from_question(
        question="Comment fonctionne un transformateur ?",
        vector_store=vector_store,
        gemini_client=gemini_client,
        settings=settings,
        mode="file_question",
    )

    vector_store.search.assert_awaited_once()
    assert result.mode == "file_question"


# --- Contrôle de couverture post-génération ---


@pytest.mark.asyncio
async def test_coverage_check_appends_missing_pages(settings, gemini_client):
    """Des pages non citées dans les sources doivent être intégrées dans des
    sections réelles via un 2e appel Gemini (complétion de couverture)."""
    vector_store = AsyncMock()
    vector_store.search.return_value = [
        {"content": "contenu page 1", "metadata": {"filename": "doc.pdf", "page": 1}, "distance": 0.1},
    ]
    vector_store.count_pages = MagicMock(return_value=2)
    vector_store.get_all_chunks = MagicMock(
        return_value=[
            {"content": "contenu page 1", "metadata": {"filename": "doc.pdf", "page": 1}, "distance": 0.0},                {"content": "contenu page 2 manquant sur les index B-tree en PostgreSQL. " * 10, "metadata": {"filename": "doc.pdf", "page": 2}, "distance": 0.0},
        ]
    )
    # 1er appel : réponse principale (cite seulement page 1)
    structured_main = VALID_STRUCTURED_ANSWER_FILE.copy()
    structured_main["sources"] = [{"type": "file_chunk", "label": "doc.pdf", "reference": "page 1"}]
    # 2e appel : complétion de couverture (section thématique réelle)
    structured_completion = {
        "sections": [
            {
                "type": "development",
                "title": "Index B-tree en PostgreSQL",
                "blocks": [],
                "subsections": [
                    {"title": "Quoi", "blocks": [{"type": "text", "text": "Les index B-tree optimisent les requêtes de range."}]},
                    {"title": "Pourquoi", "blocks": [{"type": "text", "text": "Ils accélèrent les recherches par comparaison."}]},
                    {"title": "Comment", "blocks": [{"type": "text", "text": "contenu page 2 manquant sur les index B-tree"}]},
                ],
            },
        ],
    }
    gemini_client.format_structured.side_effect = [structured_main, structured_completion]

    result = await generate_course_from_question(
        question="question",
        vector_store=vector_store,
        gemini_client=gemini_client,
        settings=settings,
        filename="doc.pdf",
    )

    assert gemini_client.format_structured.await_count == 2  # 1 principale + 1 complétion
    section_titles = [s.title for s in (result.sections or [])]
    assert any("index" in t.lower() or "b-tree" in t.lower() for t in section_titles)
    # Vérifier qu'aucune section ne porte un titre générique
    assert not any("complémentaire" in t.lower() or "non couvert" in t.lower() for t in section_titles)
    section_texts = [s.comment for s in (result.sections or [])]
    assert any("contenu page 2 manquant" in t for t in section_texts)
    assert any(s.label == "doc.pdf" and s.reference == "page 2" for s in result.sources)


@pytest.mark.asyncio
async def test_coverage_check_noop_when_all_pages_covered(settings, gemini_client):
    vector_store = AsyncMock()
    vector_store.search.return_value = [
        {"content": "contenu page 1", "metadata": {"filename": "doc.pdf", "page": 1}, "distance": 0.1},
    ]
    vector_store.count_pages = MagicMock(return_value=1)
    structured = VALID_STRUCTURED_ANSWER_FILE.copy()
    structured["sources"] = [{"type": "file_chunk", "label": "doc.pdf", "reference": "page 1"}]
    gemini_client.format_structured.return_value = structured

    result = await generate_course_from_question(
        question="question",
        vector_store=vector_store,
        gemini_client=gemini_client,
        settings=settings,
        filename="doc.pdf",
    )

    original_section_count = len(structured["sections"])
    assert len(result.sections or []) == original_section_count


# --- compute_quiz_points ---


def _make_quiz(difficulties: list[QuizDifficulty]) -> list[QuizQuestion]:
    """Génère des questions quiz factices avec les difficultés données."""
    return [
        QuizQuestion(
            question=f"Q{i+1}",
            choices=["A", "B", "C"],
            correct_indices=[0],
            difficulty=d,
            explanation="",
            requires_calculation=False,
        )
        for i, d in enumerate(difficulties)
    ]


def test_compute_quiz_points_empty():
    assert compute_quiz_points([]) == []


def _expected_points_for(difficulties: list[QuizDifficulty]) -> list[float]:
    """Calcule les points attendus pour une liste de difficultés (algorithme de référence)."""
    WEIGHTS = {QuizDifficulty.FACILE: 1.0, QuizDifficulty.NORMALE: 1.5, QuizDifficulty.DIFFICILE: 2.0}
    n = len(difficulties)
    weights = [WEIGHTS[d] for d in difficulties]
    tw = sum(weights)
    raw = [w / tw * 20.0 for w in weights]
    points = [round(r * 2) / 2 for r in raw]
    points = [max(0.5, p) for p in points]
    residual = round((20.0 - sum(points)) * 2) / 2
    if residual != 0:
        for i, d in enumerate(difficulties):
            if d in (QuizDifficulty.DIFFICILE, QuizDifficulty.NORMALE):
                points[i] = round((points[i] + residual) * 2) / 2
                break
    return points


def test_compute_quiz_points_sum_20():
    """La somme des points doit être exactement 20.0."""
    difficulties = (
        [QuizDifficulty.DIFFICILE] * 5
        + [QuizDifficulty.NORMALE] * 3
        + [QuizDifficulty.FACILE] * 2
    )  # 10 questions, distribution 50/30/20 ≈ 50/25/25
    questions = _make_quiz(difficulties)
    points = compute_quiz_points(questions)
    assert len(points) == 10
    assert abs(sum(points) - 20.0) < 1e-9


def test_compute_quiz_points_bounds():
    """Chaque point doit être >= 0.5 (borne inférieure stricte).
    La borne supérieure ~2.0 est une cible, pas une garantie absolue
    (impossible à respecter pour tous les N avec la distribution 50/25/25).
    """
    difficulties = (
        [QuizDifficulty.DIFFICILE] * 5
        + [QuizDifficulty.NORMALE] * 2
        + [QuizDifficulty.FACILE] * 3
    )
    questions = _make_quiz(difficulties)
    points = compute_quiz_points(questions)
    for p in points:
        assert p >= 0.5, f"{p} < 0.5"


def test_compute_quiz_points_steps_of_0_5():
    """Les points doivent être des multiples de 0.5."""
    difficulties = (
        [QuizDifficulty.DIFFICILE] * 5
        + [QuizDifficulty.NORMALE] * 2
        + [QuizDifficulty.FACILE] * 3
    )
    questions = _make_quiz(difficulties)
    points = compute_quiz_points(questions)
    for p in points:
        assert abs(p * 2 - round(p * 2)) < 1e-9, f"{p} n'est pas un multiple de 0.5"


def test_compute_quiz_points_non_multiple_of_4():
    """Avec N non multiple de 4 (ex: 9), la somme doit quand même être 20.0."""
    difficulties = (
        [QuizDifficulty.DIFFICILE] * 5
        + [QuizDifficulty.NORMALE] * 2
        + [QuizDifficulty.FACILE] * 2
    )  # 9 questions
    questions = _make_quiz(difficulties)
    points = compute_quiz_points(questions)
    assert abs(sum(points) - 20.0) < 1e-9
    assert len(points) == 9


def test_compute_quiz_points_all_same_difficulty():
    """Toutes les questions de même difficulté : points identiques, somme 20."""
    difficulties = [QuizDifficulty.DIFFICILE] * 10
    questions = _make_quiz(difficulties)
    points = compute_quiz_points(questions)
    assert abs(sum(points) - 20.0) < 1e-9
    assert len(set(points)) <= 2  # au plus 2 valeurs différentes (reliat ajusté)


# --- Pydantic validation — QuizQuestion LLM schema ---


def test_quiz_question_correct_indices_single():
    q = QuizQuestion(
        question="Q",
        choices=["A", "B", "C"],
        correct_indices=[1],
        difficulty=QuizDifficulty.NORMALE,
        explanation="",
        requires_calculation=False,
    )
    assert q.correct_indices == [1]


@pytest.mark.asyncio
async def test_coverage_check_noop_below_threshold(settings, gemini_client):
    """Volume manquant sous le seuil → aucun 2e appel Gemini, sections inchangées."""
    vector_store = AsyncMock()
    vector_store.search.return_value = [
        {"content": "contenu page 1", "metadata": {"filename": "doc.pdf", "page": 1}, "distance": 0.1},
    ]
    vector_store.count_pages = MagicMock(return_value=2)
    # Chunk manquant très court (< course_coverage_min_missing_chars = 300)
    vector_store.get_all_chunks = MagicMock(
        return_value=[
            {"content": "x", "metadata": {"filename": "doc.pdf", "page": 1}, "distance": 0.0},
            {"content": "court", "metadata": {"filename": "doc.pdf", "page": 2}, "distance": 0.0},
        ]
    )
    structured = VALID_STRUCTURED_ANSWER_FILE.copy()
    structured["sources"] = [{"type": "file_chunk", "label": "doc.pdf", "reference": "page 1"}]
    gemini_client.format_structured.return_value = structured

    result = await generate_course_from_question(
        question="question",
        vector_store=vector_store,
        gemini_client=gemini_client,
        settings=settings,
        filename="doc.pdf",
    )

    # Un seul appel Gemini (pas de complétion)
    assert gemini_client.format_structured.await_count == 1
    assert len(result.sections or []) == len(structured["sections"])


@pytest.mark.asyncio
async def test_coverage_check_disabled_via_settings(gemini_client):
    """course_coverage_completion_enabled=False → aucun 2e appel même avec contenu manquant."""
    disabled_settings = Settings(
        gemini_api_key="fake-key",
        gemini_use_search_grounding=True,
        course_coverage_completion_enabled=False,
    )
    vector_store = AsyncMock()
    vector_store.search.return_value = [
        {"content": "contenu page 1", "metadata": {"filename": "doc.pdf", "page": 1}, "distance": 0.1},
    ]
    vector_store.count_pages = MagicMock(return_value=2)
    vector_store.get_all_chunks = MagicMock(
        return_value=[
            {"content": "x" * 500, "metadata": {"filename": "doc.pdf", "page": 1}, "distance": 0.0},
            {"content": "y" * 500, "metadata": {"filename": "doc.pdf", "page": 2}, "distance": 0.0},
        ]
    )
    structured = VALID_STRUCTURED_ANSWER_FILE.copy()
    structured["sources"] = [{"type": "file_chunk", "label": "doc.pdf", "reference": "page 1"}]
    gemini_client.format_structured.return_value = structured

    result = await generate_course_from_question(
        question="question",
        vector_store=vector_store,
        gemini_client=gemini_client,
        settings=disabled_settings,
        filename="doc.pdf",
    )

    assert gemini_client.format_structured.await_count == 1


@pytest.mark.asyncio
async def test_coverage_completion_gemini_failure_is_non_blocking(settings, gemini_client):
    """Exception Gemini lors de la complétion → cours principal intact, aucune exception propagée."""
    vector_store = AsyncMock()
    vector_store.search.return_value = [
        {"content": "contenu page 1", "metadata": {"filename": "doc.pdf", "page": 1}, "distance": 0.1},
    ]
    vector_store.count_pages = MagicMock(return_value=2)
    vector_store.get_all_chunks = MagicMock(
        return_value=[
            {"content": "x" * 500, "metadata": {"filename": "doc.pdf", "page": 1}, "distance": 0.0},
            {"content": "y" * 500, "metadata": {"filename": "doc.pdf", "page": 2}, "distance": 0.0},
        ]
    )
    structured_main = VALID_STRUCTURED_ANSWER_FILE.copy()
    structured_main["sources"] = [{"type": "file_chunk", "label": "doc.pdf", "reference": "page 1"}]
    # 1er appel OK, 2e appel lève une exception
    gemini_client.format_structured.side_effect = [structured_main, GeminiUnavailableError("Gemini down")]

    result = await generate_course_from_question(
        question="question",
        vector_store=vector_store,
        gemini_client=gemini_client,
        settings=settings,
        filename="doc.pdf",
    )

    # Le cours principal est retourné intact
    assert result.mode == "file_question"
    assert result.sections is not None
    assert len(result.sections) == 1  # seule la section initiale, pas de complétion


@pytest.mark.asyncio
async def test_coverage_completion_merges_into_existing_section_by_title(settings, gemini_client):
    """Gemini renvoie une section avec un titre existant → fusion (pas de doublon)."""
    vector_store = AsyncMock()
    vector_store.search.return_value = [
        {"content": "contenu page 1", "metadata": {"filename": "doc.pdf", "page": 1}, "distance": 0.1},
    ]
    vector_store.count_pages = MagicMock(return_value=2)
    vector_store.get_all_chunks = MagicMock(
        return_value=[
            {"content": "x" * 500, "metadata": {"filename": "doc.pdf", "page": 1}, "distance": 0.0},
            {"content": "y" * 500, "metadata": {"filename": "doc.pdf", "page": 2}, "distance": 0.0},
        ]
    )
    structured_main = VALID_STRUCTURED_ANSWER_FILE.copy()
    structured_main["sources"] = [{"type": "file_chunk", "label": "doc.pdf", "reference": "page 1"}]
    # 2e appel : section avec même titre que la section existante
    structured_completion = {
        "sections": [
            {
                "type": "development",
                "title": "Le transformateur",  # même titre que la section initiale
                "blocks": [],
                "subsections": [
                    {"title": "Quoi", "blocks": [{"type": "text", "text": ""}]},
                    {"title": "Pourquoi", "blocks": [{"type": "text", "text": ""}]},
                    {"title": "Comment", "blocks": [{"type": "text", "text": "détails supplémentaires sur le transformateur"}]},
                ],
            },
        ],
    }
    gemini_client.format_structured.side_effect = [structured_main, structured_completion]

    result = await generate_course_from_question(
        question="question",
        vector_store=vector_store,
        gemini_client=gemini_client,
        settings=settings,
        filename="doc.pdf",
    )

    # Pas de doublon : une seule section "Le transformateur"
    titles = [s.title for s in (result.sections or [])]
    assert titles.count("Le transformateur") == 1
    # Le comment de la section existante doit contenir le contenu fusionné
    transformer_section = next(s for s in result.sections if s.title == "Le transformateur")
    assert "détails supplémentaires" in transformer_section.comment


@pytest.mark.asyncio
async def test_map_sections_to_course_sections_non_regression():
    """_map_sections_to_course_sections produit un résultat identique à l'ancien mapping inline."""
    from app.schemas.course_generation import (
        ContentBlock,
        Section,
        SectionType,
        Subsection,
    )
    sections = [
        Section(
            type=SectionType.DEVELOPMENT,
            title="Le transformateur",
            subsections=[
                Subsection(title="Quoi", blocks=[ContentBlock(type="text", text="définition")]),
                Subsection(title="Pourquoi", blocks=[ContentBlock(type="text", text="raison")]),
                Subsection(title="Comment", blocks=[ContentBlock(type="text", text="mécanisme")]),
            ],
        ),
        Section(
            type=SectionType.INTRODUCTION,
            title="Introduction",
            blocks=[ContentBlock(type="text", text="intro text")],
        ),
    ]
    api_sections = _map_sections_to_course_sections(sections)
    assert len(api_sections) == 2
    assert api_sections[0].title == "Le transformateur"
    assert api_sections[0].quoi == "définition"
    assert api_sections[0].pourquoi == "raison"
    assert api_sections[0].comment == "mécanisme"
    assert api_sections[1].title == "Introduction"
    assert api_sections[1].quoi == "intro text"


def test_quiz_question_correct_indices_multiple():
    q = QuizQuestion(
        question="Q",
        choices=["A", "B", "C", "D"],
        correct_indices=[0, 2],
        difficulty=QuizDifficulty.DIFFICILE,
        explanation="",
        requires_calculation=False,
    )
    assert q.correct_indices == [0, 2]


def test_quiz_question_correct_indices_empty_rejected():
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match="correct_indices ne peut pas être vide"):
        QuizQuestion(
            question="Q",
            choices=["A", "B"],
            correct_indices=[],
            difficulty=QuizDifficulty.FACILE,
            explanation="",
            requires_calculation=False,
        )


def test_quiz_question_correct_indices_duplicates_rejected():
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match="doublons"):
        QuizQuestion(
            question="Q",
            choices=["A", "B", "C"],
            correct_indices=[0, 0],
            difficulty=QuizDifficulty.FACILE,
            explanation="",
            requires_calculation=False,
        )


def test_quiz_question_correct_indices_out_of_bounds_rejected():
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match="hors bornes"):
        QuizQuestion(
            question="Q",
            choices=["A", "B"],
            correct_indices=[5],
            difficulty=QuizDifficulty.FACILE,
            explanation="",
            requires_calculation=False,
        )


# --- Quiz mapping dans generate_course_from_question ---


def _quiz_entries(n_diff: int, n_norm: int, n_fac: int) -> list[dict]:
    """Génère des entrées quiz factices avec la distribution donnée."""
    entries = []
    idx = 0
    for diff, count in [("difficile", n_diff), ("normale", n_norm), ("facile", n_fac)]:
        for _ in range(count):
            idx += 1
            entries.append({
                "question": f"Q{idx}",
                "choices": ["A", "B", "C", "D"],
                "correct_indices": [0],
                "difficulty": diff,
                "explanation": "",
                "requires_calculation": diff == "difficile",
            })
    return entries


VALID_STRUCTURED_ANSWER_WITH_QUIZ = {
    "mode": "file_question",
    "format": "focused_answer",
    "meta": {
        "title": "Algebre",
        "subject": "Maths",
        "language": "fr",
        "generated_at": "2026-08-19T10:00:00Z",
    },
    "sources": [{"type": "file_chunk", "label": "doc.pdf", "reference": "doc_1_chunk_0"}],
    "sections": [
        {
            "type": "development",
            "title": "Les matrices",
            "blocks": [],
            "subsections": [
                {"title": "Quoi", "blocks": [{"type": "text", "text": "definition"}]},
                {"title": "Pourquoi", "blocks": [{"type": "text", "text": "raison"}]},
                {"title": "Comment", "blocks": [{"type": "text", "text": "mecanisme"}]},
            ],
        },
    ],
    "quiz": _quiz_entries(n_diff=5, n_norm=2, n_fac=3),
    "confidence": "high",
    "unconfirmed_points": [],
}


@pytest.mark.asyncio
async def test_generate_course_quiz_has_points_and_indices(settings, gemini_client):
    """Le quiz映射é doit contenir correct_option_indices, difficulty, points."""
    vector_store = AsyncMock()
    vector_store.search.return_value = [
        {"content": "extrait", "metadata": {"filename": "doc.pdf", "page": 1}, "distance": 0.1},
    ]
    vector_store.count_pages = MagicMock(return_value=0)
    gemini_client.format_structured.return_value = VALID_STRUCTURED_ANSWER_WITH_QUIZ.copy()

    result = await generate_course_from_question(
        question="question",
        vector_store=vector_store,
        gemini_client=gemini_client,
        settings=settings,
    )

    assert result.quiz is not None
    assert len(result.quiz) == 10
    total_points = 0.0
    for q in result.quiz:
        assert hasattr(q, "correct_option_indices")
        assert hasattr(q, "difficulty")
        assert hasattr(q, "points")
        assert q.points >= 0.5
        assert q.difficulty in ("facile", "normale", "difficile")
        assert len(q.correct_option_indices) >= 1
        total_points += q.points
    assert abs(total_points - 20.0) < 1e-9


@pytest.mark.asyncio
async def test_generate_course_quiz_single_answer(settings, gemini_client):
    """Question à réponse unique : correct_option_indices a un seul élément."""
    structured = VALID_STRUCTURED_ANSWER_WITH_QUIZ.copy()
    structured["quiz"] = [
        {
            "question": "Q unique",
            "choices": ["A", "B", "C"],
            "correct_indices": [1],
            "difficulty": "normale",
            "explanation": "",
            "requires_calculation": False,
        },
    ]
    vector_store = AsyncMock()
    vector_store.search.return_value = [
        {"content": "extrait", "metadata": {"filename": "doc.pdf", "page": 1}, "distance": 0.1},
    ]
    vector_store.count_pages = MagicMock(return_value=0)
    gemini_client.format_structured.return_value = structured

    result = await generate_course_from_question(
        question="question",
        vector_store=vector_store,
        gemini_client=gemini_client,
        settings=settings,
    )

    assert result.quiz is not None
    assert len(result.quiz) == 1
    assert result.quiz[0].correct_option_indices == [1]


@pytest.mark.asyncio
async def test_generate_course_quiz_multiple_answers(settings, gemini_client):
    """Question à réponses multiples : correct_option_indices a 2+ éléments."""
    structured = VALID_STRUCTURED_ANSWER_WITH_QUIZ.copy()
    structured["quiz"] = [
        {
            "question": "Q multi",
            "choices": ["A", "B", "C", "D"],
            "correct_indices": [0, 2, 3],
            "difficulty": "difficile",
            "explanation": "",
            "requires_calculation": True,
        },
    ]
    vector_store = AsyncMock()
    vector_store.search.return_value = [
        {"content": "extrait", "metadata": {"filename": "doc.pdf", "page": 1}, "distance": 0.1},
    ]
    vector_store.count_pages = MagicMock(return_value=0)
    gemini_client.format_structured.return_value = structured

    result = await generate_course_from_question(
        question="question",
        vector_store=vector_store,
        gemini_client=gemini_client,
        settings=settings,
    )

    assert result.quiz is not None
    assert result.quiz[0].correct_option_indices == [0, 2, 3]
    assert result.quiz[0].difficulty == "difficile"


@pytest.mark.asyncio
async def test_generate_course_empty_quiz_still_works(settings, gemini_client):
    """Quiz vide doit toujours fonctionner (non-régression)."""
    gemini_client.format_structured.return_value = VALID_STRUCTURED_ANSWER_FILE.copy()
    vector_store = AsyncMock()
    vector_store.search.return_value = [
        {"content": "extrait", "metadata": {"filename": "doc.pdf", "page": 1}, "distance": 0.1},
    ]
    vector_store.count_pages = MagicMock(return_value=0)

    result = await generate_course_from_question(
        question="question",
        vector_store=vector_store,
        gemini_client=gemini_client,
        settings=settings,
    )

    assert result.quiz is None or result.quiz == []


# --- Fix répartition difficulté quiz (Partie C) ---


def test_is_quiz_difficulty_error_detects():
    assert _is_quiz_difficulty_error(ValueError("Répartition de difficulté incorrecte : 5")) is True
    assert _is_quiz_difficulty_error(ValueError("quiz_difficulty distribution")) is True
    assert _is_quiz_difficulty_error(ValueError("autre erreur")) is False
    assert _is_quiz_difficulty_error(ValueError("")) is False


def test_rebalance_quiz_difficulty_n14():
    """N=14 → difficile=7, normale=4, facile=3."""
    quiz = [{"question": f"Q{i}", "choices": ["A", "B"], "correct_indices": [0],
             "difficulty": "facile", "explanation": "", "requires_calculation": False}
            for i in range(14)]
    structured = {"quiz": quiz}
    result = _rebalance_quiz_difficulty(structured)
    counts = {"difficile": 0, "normale": 0, "facile": 0}
    for q in result["quiz"]:
        counts[q["difficulty"]] += 1
    assert counts["difficile"] == 7
    assert counts["normale"] == 4
    assert counts["facile"] == 3


def test_rebalance_quiz_difficulty_n12():
    """N=12 → difficile=6, normale=3, facile=3."""
    quiz = [{"question": f"Q{i}", "choices": ["A", "B"], "correct_indices": [0],
             "difficulty": "normale", "explanation": "", "requires_calculation": False}
            for i in range(12)]
    structured = {"quiz": quiz}
    result = _rebalance_quiz_difficulty(structured)
    counts = {"difficile": 0, "normale": 0, "facile": 0}
    for q in result["quiz"]:
        counts[q["difficulty"]] += 1
    assert counts["difficile"] == 6
    assert counts["normale"] == 3
    assert counts["facile"] == 3


def test_rebalance_quiz_difficulty_n16():
    """N=16 → difficile=8, normale=4, facile=4."""
    quiz = [{"question": f"Q{i}", "choices": ["A", "B"], "correct_indices": [0],
             "difficulty": "difficile", "explanation": "", "requires_calculation": False}
            for i in range(16)]
    structured = {"quiz": quiz}
    result = _rebalance_quiz_difficulty(structured)
    counts = {"difficile": 0, "normale": 0, "facile": 0}
    for q in result["quiz"]:
        counts[q["difficulty"]] += 1
    assert counts["difficile"] == 8
    assert counts["normale"] == 4
    assert counts["facile"] == 4


def test_rebalance_quiz_difficulty_n2():
    """N=2 → difficile=1, normale=0 (round(0.5)=0 banker), facile=1."""
    quiz = [{"question": f"Q{i}", "choices": ["A", "B"], "correct_indices": [0],
             "difficulty": "facile", "explanation": "", "requires_calculation": False}
            for i in range(2)]
    structured = {"quiz": quiz}
    result = _rebalance_quiz_difficulty(structured)
    counts = {"difficile": 0, "normale": 0, "facile": 0}
    for q in result["quiz"]:
        counts[q["difficulty"]] += 1
    assert counts["difficile"] == round(2 / 2)
    assert counts["normale"] == round(2 / 4)
    assert counts["facile"] == 2 - counts["difficile"] - counts["normale"]


def test_rebalance_quiz_difficulty_n3():
    """N=3 → difficile=2, normale=1, facile=0."""
    quiz = [{"question": f"Q{i}", "choices": ["A", "B"], "correct_indices": [0],
             "difficulty": "facile", "explanation": "", "requires_calculation": False}
            for i in range(3)]
    structured = {"quiz": quiz}
    result = _rebalance_quiz_difficulty(structured)
    counts = {"difficile": 0, "normale": 0, "facile": 0}
    for q in result["quiz"]:
        counts[q["difficulty"]] += 1
    assert counts["difficile"] == 2
    assert counts["normale"] == 1
    assert counts["facile"] == 0


def test_rebalance_quiz_difficulty_n20():
    """N=20 → difficile=10, normale=5, facile=5."""
    quiz = [{"question": f"Q{i}", "choices": ["A", "B"], "correct_indices": [0],
             "difficulty": "normale", "explanation": "", "requires_calculation": False}
            for i in range(20)]
    structured = {"quiz": quiz}
    result = _rebalance_quiz_difficulty(structured)
    counts = {"difficile": 0, "normale": 0, "facile": 0}
    for q in result["quiz"]:
        counts[q["difficulty"]] += 1
    assert counts["difficile"] == 10
    assert counts["normale"] == 5
    assert counts["facile"] == 5


def test_rebalance_quiz_difficulty_empty_noop():
    """Quiz vide ou None → pas de modification."""
    assert _rebalance_quiz_difficulty({}) == {}
    assert _rebalance_quiz_difficulty({"quiz": []}) == {"quiz": []}
    assert _rebalance_quiz_difficulty({"quiz": None}) == {"quiz": None}


def test_rebalance_quiz_difficulty_n1_noop():
    """N=1 → pas de rééquilibrage (trop peu de questions)."""
    quiz = [{"question": "Q1", "choices": ["A", "B"], "correct_indices": [0],
             "difficulty": "facile", "explanation": "", "requires_calculation": False}]
    structured = {"quiz": quiz}
    result = _rebalance_quiz_difficulty(structured)
    assert result["quiz"][0]["difficulty"] == "facile"  # inchangé


def test_validate_and_map_with_bad_distribution_rebalances():
    """Quiz avec 5 difficile (N=14, attendu 7) → _validate_and_map doit rebalancer."""
    # Construire un structured avec une distribution incorrecte
    structured = VALID_STRUCTURED_ANSWER_FILE.copy()
    structured["quiz"] = [
        {"question": f"Q{i}", "choices": ["A", "B", "C"], "correct_indices": [0],
         "difficulty": "difficile", "explanation": "", "requires_calculation": False}
        for i in range(5)
    ] + [
        {"question": f"Q{i}", "choices": ["A", "B", "C"], "correct_indices": [0],
         "difficulty": "difficile", "explanation": "", "requires_calculation": False}
        for i in range(5, 9)
    ] + [
        {"question": f"Q{i}", "choices": ["A", "B", "C"], "correct_indices": [0],
         "difficulty": "facile", "explanation": "", "requires_calculation": False}
        for i in range(9, 14)
    ]
    # 14 questions, 9 difficile → déclenche l'erreur
    result = _validate_and_map(structured, "file_question")
    assert result.quiz is not None
    assert len(result.quiz) == 14
    # Vérifier que la distribution est maintenant valide (attributs, pas subscripts)
    counts = {"difficile": 0, "normale": 0, "facile": 0}
    for q in result.quiz:            counts[q.difficulty] += 1
    expected_d = round(14 / 2)
    expected_n = round(14 / 4)
    expected_f = 14 - expected_d - expected_n
    assert counts["difficile"] == expected_d
    assert counts["normale"] == expected_n
    assert counts["facile"] == expected_f