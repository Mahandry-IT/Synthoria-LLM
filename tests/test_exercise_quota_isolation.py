"""Non-régression de quota : générer un cours (ou son quiz) n'appelle jamais les exercices (RG-4, RG-5)."""

import ast
from pathlib import Path

import app

APP_DIR = Path(app.__file__).parent
# Modules du parcours « génération de cours / quiz / chat / podcast » : aucun ne doit importer les exercices.
COURSE_PATH_MODULES = [
    "api/routes.py", "api/quiz_routes.py", "api/chat_routes.py", "api/podcast_routes.py",
    "services/course_generator.py", "services/course_plan_generator.py", "services/course_regenerator.py",
    "services/course_section_adder.py", "services/quiz_bank.py", "workers/podcast_worker.py",
]


def _imported_modules(path: Path) -> set[str]:
    """Noms complets des modules importés, y compris les `from pkg import module` (pkg.module)."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{a.name}" for a in node.names)
    return names


def test_course_generation_path_never_imports_exercises():
    offenders = {
        m: sorted(n for n in _imported_modules(APP_DIR / m) if "exercise" in n.lower())
        for m in COURSE_PATH_MODULES
    }
    assert {m: n for m, n in offenders.items() if n} == {}
