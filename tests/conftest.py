"""Shared fixtures: an adapter on a throwaway AGE graph.

The connection settings come from the environment, with the defaults of the
local container in the README:

* ``AGE_HOST`` and ``AGE_PORT`` name the PostgreSQL server.
* ``AGE_USER`` and ``AGE_PASSWORD`` name the graph role. 🚨 It is never a
  superuser: ``test_role.py`` fails the run if it is one.
* ``AGE_DATABASE`` names the graph database, where the extension exists.
* ``AGE_REQUIRED`` is ``1`` in CI. 🚨 A skipped gate is a green gate. Without a
  server the fixtures skip, which is right on a laptop and a false pass in CI.
  With this flag set, an unreachable server fails the run instead.
"""

from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from typing import NoReturn

import psycopg
import pytest

from cognee_age_adapter import AGEAdapter

HOST = os.getenv("AGE_HOST", "127.0.0.1")
PORT = int(os.getenv("AGE_PORT", "5432"))
USER = os.getenv("AGE_USER", "cognee_graph")
PASSWORD = os.getenv("AGE_PASSWORD", "cognee_graph")
DATABASE = os.getenv("AGE_DATABASE", "cognee_graph")
REQUIRE_SERVER = os.getenv("AGE_REQUIRED", "").strip().lower() in {"1", "true", "yes"}


def connect() -> psycopg.Connection:
    """A plain connection as the graph role, for test setup and assertions."""
    return psycopg.connect(
        host=HOST, port=PORT, user=USER, password=PASSWORD, dbname=DATABASE, autocommit=True
    )


def unreachable(exc: Exception) -> NoReturn:
    message = f"AGE not reachable at {HOST}:{PORT}/{DATABASE} as {USER}: {exc}"
    if REQUIRE_SERVER:
        pytest.fail(message)
    pytest.skip(message)


def drop_graph(name: str) -> None:
    with connect() as conn:
        try:
            conn.execute("SELECT ag_catalog.drop_graph(%s, true)", (name,))
        except psycopg.errors.InvalidSchemaName:
            pass  # the graph was never created


def new_adapter(graph_name: str) -> AGEAdapter:
    return AGEAdapter(
        host=HOST,
        port=PORT,
        graph_database_username=USER,
        graph_database_password=PASSWORD,
        postgres_database=DATABASE,
        graph_database_name=graph_name,
    )


def record_sql(adapter: AGEAdapter) -> list:
    """Record every ``(sql, args)`` that ``adapter`` sends from now on.

    Wraps ``_execute``, the one place the adapter sends SQL, so the recorded
    statement is exactly what PostgreSQL receives. ``del adapter._execute``
    stops the recording.
    """
    sent: list = []
    original = adapter._execute

    async def recording(cursor, sql, args):
        sent.append((sql, args))
        await original(cursor, sql, args)

    adapter._execute = recording
    return sent


@asynccontextmanager
async def throwaway_adapter(prefix: str):
    """An initialized adapter on a new graph, dropped afterwards."""
    graph_name = f"{prefix}_{uuid.uuid4().hex[:12]}"
    adapter = new_adapter(graph_name)
    try:
        # 🚨 Nothing else creates the labels and the indexes. cognee calls this
        # when it builds the engine; the fixture stands in for that.
        await adapter.initialize()
    except (psycopg.OperationalError, OSError) as exc:
        await adapter.close()
        unreachable(exc)
    try:
        yield adapter
    finally:
        await adapter.close()
        drop_graph(graph_name)


@pytest.fixture
async def adapter():
    async with throwaway_adapter("test") as instance:
        yield instance
