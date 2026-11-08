"""cognee's factory can build this adapter, with the settings it is given.

Every other test constructs ``AGEAdapter`` directly, the contract suite too, so
none of them exercises the one call that matters in production:

    adapter = supported_databases[graph_database_provider]
    return adapter(graph_database_url=…, graph_database_username=…,
                   graph_database_password=…, graph_database_port=…,
                   graph_database_key=…, database_name=…)

🚨 So the keyword list is **read out of cognee's AST**, not restated here. A
restated list is how the FalkorDB adapter once shipped with a constructor that
cognee could not call (a ``TypeError`` on the first cognify).

The server-free tests record the connection settings at the pool, which is the
boundary between the adapter and PostgreSQL.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
import uuid
from importlib import import_module

import pytest
from psycopg.conninfo import conninfo_to_dict
from psycopg_pool import AsyncConnectionPool

from cognee.infrastructure.databases.graph.supported_databases import supported_databases

# ⚠ Imported by path, not `from …graph import get_graph_engine`: that package's
# __init__ re-exports the *function* under the module's own name, so the plain
# form binds a function and the AST walk reads the wrong source.
engine_module = import_module("cognee.infrastructure.databases.graph.get_graph_engine")

from cognee_age_adapter import AGEAdapter
from cognee_age_adapter.constants import DEFAULT_GRAPH_NAME
from cognee_age_adapter.register import PROVIDER_NAME, ensure_registered

import conftest


class _RecordingPool:
    """Stands in for ``psycopg_pool.AsyncConnectionPool``; records how it was built."""

    last: "_RecordingPool | None" = None
    check_connection = AsyncConnectionPool.check_connection

    def __init__(self, conninfo, **kwargs):
        self.conninfo = conninfo
        self.kwargs = kwargs
        type(self).last = self

    @property
    def settings(self) -> dict:
        return conninfo_to_dict(self.conninfo)


@pytest.fixture
def recording_pool(monkeypatch):
    monkeypatch.setattr("cognee_age_adapter.adapter.AsyncConnectionPool", _RecordingPool)
    _RecordingPool.last = None
    return _RecordingPool


def _registered_adapter_kwargs() -> set[str]:
    """The keywords cognee passes a registered adapter, parsed from its own source."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(engine_module._create_graph_engine)))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "adapter"
    ]
    assert len(calls) == 1, (
        f"expected exactly one `adapter(...)` construction in cognee's "
        f"_create_graph_engine, found {len(calls)} — the registered-provider "
        "dispatch has been restructured; re-read it before trusting this test"
    )
    return {keyword.arg for keyword in calls[0].keywords if keyword.arg is not None}


def test_constructor_binds_every_keyword_cognee_passes():
    kwargs = _registered_adapter_kwargs()
    assert kwargs, "parsed no keywords — the AST walk is looking at the wrong call"

    # bind() raises TypeError on an unexpected or missing keyword, which is the
    # exact failure production would hit, one layer earlier.
    inspect.signature(AGEAdapter).bind(**dict.fromkeys(kwargs))


def test_registration_binds_the_provider_name():
    ensure_registered()
    assert supported_databases[PROVIDER_NAME] is AGEAdapter


def test_registration_refuses_a_vector_store_other_than_pgvector(monkeypatch):
    monkeypatch.setenv("VECTOR_DB_PROVIDER", "falkordb")
    with pytest.raises(RuntimeError, match="GRAPH adapter only"):
        ensure_registered()


def test_factory_dispatch_reaches_this_adapter(recording_pool):
    """Through cognee's own dispatch, with the production shape of the arguments.

    ``graph_database_url`` carries a bare host, and the port arrives as a string,
    because cognee stringifies it on the way in. One name, ``cognee_graph``, is
    the AGE graph and the PostgreSQL database.
    """
    ensure_registered()
    adapter = supported_databases["age"](
        graph_database_url="postgres.lan",
        graph_database_username="cognee_graph",
        graph_database_password="secret",
        graph_database_port="5432",
        graph_database_key="",
        database_name="cognee_graph",
    )

    assert isinstance(adapter, AGEAdapter)
    settings = recording_pool.last.settings
    assert settings["host"] == "postgres.lan"
    assert settings["port"] == "5432"
    assert settings["user"] == "cognee_graph"
    assert settings["password"] == "secret"
    assert settings["dbname"] == "cognee_graph"
    assert adapter._graph_name == "cognee_graph"


def test_empty_strings_fall_back_to_defaults(recording_pool):
    """cognee passes "" for anything unset, so falsiness is what "unset" means."""
    adapter = AGEAdapter(
        graph_database_url="",
        graph_database_username="",
        graph_database_password="",
        graph_database_port="",
        graph_database_key="",
        database_name="",
    )

    settings = recording_pool.last.settings
    assert settings["host"] == "localhost"
    assert settings["port"] == "5432"
    assert settings["dbname"] == DEFAULT_GRAPH_NAME
    assert "user" not in settings and "password" not in settings
    assert adapter._graph_name == DEFAULT_GRAPH_NAME


def test_unset_port_sentinel_is_passed_through_not_defaulted(recording_pool):
    """cognee's unset port is 123. A refused connection on port 123 names the
    mistake; a silent rewrite to 5432 would connect to another server."""
    AGEAdapter(graph_database_url="postgres.lan", graph_database_port=123)
    assert recording_pool.last.settings["port"] == "123"


def test_a_url_keeps_its_own_database_and_the_graph_name_stays_separate(recording_pool):
    adapter = AGEAdapter(
        graph_database_url="postgresql://u:p@postgres.lan:6000/graphs", database_name="cognee_graph"
    )

    settings = recording_pool.last.settings
    assert (settings["host"], settings["port"], settings["dbname"]) == ("postgres.lan", "6000", "graphs")
    assert (settings["user"], settings["password"]) == ("u", "p")
    assert adapter._graph_name == "cognee_graph"


def test_an_explicit_postgres_database_wins(recording_pool):
    AGEAdapter(host="h", postgres_database="graphs", graph_database_name="contract_x1")
    assert recording_pool.last.settings["dbname"] == "graphs"


def test_no_connection_setting_loads_age_or_sets_a_search_path(recording_pool):
    """🚨 The role's own settings preload AGE. A client ``LOAD 'age'`` fails for a
    non-superuser, and an ``options=-c search_path`` would hide a role that
    lacks the setting."""
    AGEAdapter(host="h")
    assert "options" not in recording_pool.last.settings


def test_connections_are_autocommit_and_checked(recording_pool):
    """Each statement is its own transaction unless a write opens one, and a
    connection from before a server restart is replaced, not handed out."""
    AGEAdapter(host="h")
    kwargs = recording_pool.last.kwargs
    assert kwargs["kwargs"] == {"autocommit": True}
    assert kwargs["check"] is not None
    assert kwargs["open"] is False


@pytest.mark.parametrize("name", ["ab", "1graph", "graph-name", "g'); DROP TABLE x; --"])
def test_an_invalid_graph_name_is_refused(recording_pool, name):
    with pytest.raises(ValueError, match="graph name"):
        AGEAdapter(host="h", database_name=name)


def test_graph_database_key_is_accepted_and_warned_about(recording_pool, caplog):
    """PostgreSQL has no such credential, but cognee passes it always. It must
    never raise, or the adapter cannot be built for a value cognee always sends."""
    with caplog.at_level("WARNING"):
        AGEAdapter(host="h", graph_database_key="some-token")
    assert any("graph_database_key" in record.message for record in caplog.records)

    caplog.clear()
    with caplog.at_level("WARNING"):
        AGEAdapter(host="h", graph_database_key="")
    assert not caplog.records, "an empty key is the normal case and must stay quiet"


# ----------------------------------------------------------------------
# The same path, against a real server
# ----------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cognee_factory_builds_a_working_indexed_engine():
    """``create_graph_engine`` → a connected adapter whose ``initialize()`` indexes."""
    ensure_registered()
    graph_name = f"factory_{uuid.uuid4().hex[:12]}"
    # A URL names the test database; with a bare host the database would be
    # the graph's own name, as in production.
    engine = engine_module.create_graph_engine(
        graph_database_provider="age",
        graph_file_path=None,
        graph_database_url=f"postgresql://{conftest.HOST}:{conftest.PORT}/{conftest.DATABASE}",
        graph_database_username=conftest.USER,
        graph_database_password=conftest.PASSWORD,
        graph_database_name=graph_name,
    )
    assert isinstance(engine, AGEAdapter)
    try:
        try:
            await engine.initialize()
        except OSError as exc:
            conftest.unreachable(exc)
        indexes = {
            row[0]
            for row in await engine._sql(
                "SELECT indexname FROM pg_indexes WHERE schemaname = %s", (graph_name,)
            )
        }
        assert {"node_id_btree", "edge_identity"} <= indexes
    finally:
        await engine.close()
        conftest.drop_graph(graph_name)
