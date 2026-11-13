"""Données et fabriques communes aux tests des exercices corrigés."""

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

TEXT_BLOCK = {"type": "text", "text": "Calcule la résistance équivalente de deux résistances de 10 et 20 ohms."}


def draft(**overrides) -> dict:
    base = {
        "section_id": "s1",
        "title": "Résistance équivalente",
        "covers": "Associer des résistances en série.",
        "statement": [TEXT_BLOCK],
        "hints": ["Pense à la loi des mailles.", "Les résistances sont en série.", "Additionne-les."],
        "duration_minutes": 10,
        "expected_solution": "R = 10 + 20 = 30 ohms.",
    }
    base.update(overrides)
    return base


def course(depth: str = "standard", *, sections=None, quiz=True) -> dict:
    """Cours persisté minimal : deux sections complètes, un quiz valide."""
    def section(sid: str, title: str) -> dict:
        block = {"type": "text", "text": "Contenu."}
        return {
            "id": sid, "title": title, "quoi": "", "pourquoi": "", "comment": "",
            "key_takeaways": ["Notion A", "Notion B"],
            "subsections": [{"title": t, "blocks": [block]} for t in ("Pourquoi", "Quoi", "Comment")],
        }

    return {
        "meta": {"title": "Électricité", "depth": depth},
        "sections": sections if sections is not None else [section("s1", "Résistances"), section("s2", "Histoire")],
        "quiz": [
            {
                "question": "Q ?", "options": ["a", "b"], "correct_option_indices": [0], "difficulty": "normale",
                "explanation": "e", "requires_calculation": False,
            }
        ] if quiz else [],
    }


def exercise_row(**overrides):
    now = datetime.now(timezone.utc)
    fields = dict(
        id=uuid.uuid4(), session_id=uuid.uuid4(), section_id="s1", position=1, title="Titre", covers="Couvre X",
        statement=[TEXT_BLOCK], hints=["a", "b", "c"], duration_minutes=10,
        expected_solution="SOLUTION-SECRETE", created_at=now, updated_at=now,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def job_row(status: str = "pending", **overrides):
    now = datetime.now(timezone.utc)
    fields = dict(
        id=uuid.uuid4(), session_id=uuid.uuid4(), status=status, params_hash="h" * 64, attempts=0,
        locked_at=None, error_message=None, created_at=now, updated_at=now,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)
