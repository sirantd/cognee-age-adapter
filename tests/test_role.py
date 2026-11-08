"""The gate runs as a production-shaped graph role, or it proves less than it says.

🚨 homelab#332 requires that no client sends ``LOAD 'age'``: for a non-superuser
it fails with ``access to library "age" is not allowed``. Every other test only
proves that if it connects as a non-superuser whose AGE comes from the role's
own ``session_preload_libraries``. A gate that silently runs as ``postgres``
would pass a client that sends ``LOAD``.
"""

from __future__ import annotations

import uuid

import psycopg
import pytest

import conftest

pytestmark = [pytest.mark.integration]


@pytest.fixture
def conn():
    try:
        connection = conftest.connect()
    except psycopg.OperationalError as exc:
        conftest.unreachable(exc)
    with connection:
        yield connection


def test_the_graph_role_is_not_a_superuser(conn):
    row = conn.execute("SELECT rolsuper FROM pg_roles WHERE rolname = current_user").fetchone()
    assert row == (False,), (
        f"the tests connect as a superuser ({conftest.USER}): the gate proves nothing about LOAD"
    )


def test_the_graph_role_cannot_load_age(conn):
    with pytest.raises(psycopg.errors.InsufficientPrivilege, match='access to library "age"'):
        conn.execute("LOAD 'age'")


def test_cypher_runs_on_a_fresh_connection_without_a_load(conn):
    """``cypher()`` runs only after AGE is loaded, so this proves the role preload."""
    graph = f"role_{uuid.uuid4().hex[:12]}"
    conn.execute("SELECT ag_catalog.create_graph(%s)", (graph,))
    try:
        rows = conn.execute(
            f"SELECT * FROM ag_catalog.cypher('{graph}', $$ RETURN 1 $$) AS (v ag_catalog.agtype)"
        ).fetchall()
        assert rows == [("1",)]
    finally:
        conn.execute("SELECT ag_catalog.drop_graph(%s, true)", (graph,))


def test_the_extension_is_the_pinned_release(conn):
    assert conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'age'").fetchone() == (
        "1.8.0",
    )
