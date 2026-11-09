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


def test_the_role_puts_ag_catalog_on_the_search_path(conn):
    """🚨 Load-bearing. Measured: with ``search_path=public``, ``initialize()``
    fails in AGE's own label DDL (``operator class "graphid_ops" does not
    exist``). The adapter sets no search path, so the role must."""
    assert conn.execute("SHOW search_path").fetchone()[0].startswith("ag_catalog")


def test_the_role_has_a_statement_timeout(conn):
    """homelab#332 story 6: one long query cannot freeze the store for all clients."""
    assert conn.execute("SHOW statement_timeout").fetchone()[0] not in ("0", "0ms")


def test_the_role_has_an_idle_in_transaction_timeout(conn):
    """homelab#332: a dead client cannot keep the locks of its open transaction."""
    setting = conn.execute("SHOW idle_in_transaction_session_timeout").fetchone()[0]
    assert setting not in ("0", "0ms")


def test_a_long_graph_query_stops_at_the_statement_timeout(conn):
    graph = f"role_{uuid.uuid4().hex[:12]}"
    conn.execute("SELECT ag_catalog.create_graph(%s)", (graph,))
    try:
        conn.execute("SET statement_timeout = '200ms'")
        with pytest.raises(psycopg.errors.QueryCanceled):
            conn.execute(
                f"SELECT * FROM ag_catalog.cypher('{graph}', "
                # Two small lists, 10^10 rows: ``range(1, 10^9)`` fails on
                # memory first, because AGE builds the whole list.
                "$$ UNWIND range(1, 100000) AS a UNWIND range(1, 100000) AS b "
                "RETURN count(b) $$) AS (v ag_catalog.agtype)"
            )
    finally:
        conn.execute("RESET statement_timeout")
        conn.execute("SELECT ag_catalog.drop_graph(%s, true)", (graph,))


def test_the_extension_is_the_pinned_release(conn):
    assert conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'age'").fetchone() == (
        "1.8.0",
    )
