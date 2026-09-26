"""merge course_sessions folder/subfolder rows that only differ by casing

Revision ID: 012
Revises: 011
Create Date: 2026-09-26
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = "012"
down_revision = "011"
branch_labels = None
depends_on = None


def _merge_casing(conn: sa.engine.Connection, *, folder: str | None) -> None:
    """Fusionne les orthographes qui ne diffèrent que par la casse, dans `course_sessions.folder`
    (si `folder` est None) ou dans `course_sessions.subfolder` au sein d'un dossier donné.

    Casse canonique = celle du plus grand nombre de cours ; égalité départagée par la ligne la plus
    ancienne (`MIN(created_at)`)."""
    column = "folder" if folder is None else "subfolder"
    where_folder = "" if folder is None else "WHERE folder = :folder"
    params = {} if folder is None else {"folder": folder}

    groups = conn.execute(
        sa.text(
            f"""
            SELECT lower({column}) AS lowered, {column} AS spelling, COUNT(*) AS n, MIN(created_at) AS first
            FROM course_sessions
            {where_folder}
            GROUP BY lower({column}), {column}
            """
        ),
        params,
    ).all()

    by_lowered: dict[str, list[sa.engine.Row]] = {}
    for row in groups:
        by_lowered.setdefault(row.lowered, []).append(row)

    for lowered, spellings in by_lowered.items():
        if len(spellings) <= 1:
            continue
        canonical = max(spellings, key=lambda r: (r.n, -(r.first.timestamp())))
        others = [r.spelling for r in spellings if r.spelling != canonical.spelling]
        update_where = f"lower({column}) = :lowered AND {column} != :canonical"
        update_params = {"lowered": lowered, "canonical": canonical.spelling}
        if folder is not None:
            update_where += " AND folder = :folder"
            update_params["folder"] = folder
        conn.execute(
            sa.text(f"UPDATE course_sessions SET {column} = :canonical WHERE {update_where}"),
            update_params,
        )


def upgrade() -> None:
    conn = op.get_bind()

    _merge_casing(conn, folder=None)

    folders = conn.execute(sa.text("SELECT DISTINCT folder FROM course_sessions")).scalars().all()
    for folder in folders:
        _merge_casing(conn, folder=folder)


def downgrade() -> None:
    # Fusion de données non réversible : la répartition des casses avant fusion n'est pas conservée.
    pass
