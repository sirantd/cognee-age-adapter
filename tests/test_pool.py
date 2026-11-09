"""The connection pool after a failure or a close.

The shared PostgreSQL app restarts for its own maintenance. An adapter that
starts while the server is down must work when the server is back, and a
closed adapter must work again on its next call.

🚨 psycopg_pool cannot open a pool again after a close, and its ``PoolClosed``
is a ``psycopg.OperationalError``. The calls after the failure are therefore
not wrapped in ``conftest.unreachable``: that wrapper would turn the bug into a
skip on a laptop.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid

import psycopg
import pytest

import cognee_age_adapter.adapter as adapter_module
from cognee_age_adapter import AGEAdapter

import conftest

pytestmark = pytest.mark.integration

COUNT = "MATCH (n) RETURN count(n) AS count"


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    with contextlib.suppress(ConnectionError):
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    writer.close()


class _Relay:
    """A TCP relay to the AGE server. While it is down, it closes every
    connection that it accepts, as a server that restarts does."""

    def __init__(self) -> None:
        self.up = False
        self.server: asyncio.Server | None = None

    async def start(self) -> int:
        self.server = await asyncio.start_server(self._relay, "127.0.0.1", 0)
        return self.server.sockets[0].getsockname()[1]

    async def _relay(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if not self.up:
            writer.close()
            return
        upstream_reader, upstream_writer = await asyncio.open_connection(
            conftest.HOST, conftest.PORT
        )
        await asyncio.gather(_pipe(reader, upstream_writer), _pipe(upstream_reader, writer))

    async def stop(self) -> None:
        if self.server is not None:
            self.server.close()


async def test_a_closed_adapter_works_again_on_its_next_call(adapter):
    await adapter.close()

    assert await adapter.query(COUNT) == [{"count": 0}]


async def test_an_adapter_that_starts_while_the_server_is_down_works_when_it_is_back(
    monkeypatch,
):
    try:
        conftest.connect().close()
    except (psycopg.OperationalError, OSError) as exc:
        conftest.unreachable(exc)
    monkeypatch.setattr(adapter_module, "_POOL_OPEN_TIMEOUT", 1.0)
    relay = _Relay()
    graph = f"test_{uuid.uuid4().hex[:12]}"
    port = await relay.start()
    adapter = AGEAdapter(
        host="127.0.0.1",
        port=port,
        graph_database_username=conftest.USER,
        graph_database_password=conftest.PASSWORD,
        postgres_database=conftest.DATABASE,
        graph_database_name=graph,
    )
    try:
        with pytest.raises(psycopg.OperationalError):
            await adapter.initialize()

        relay.up = True
        await adapter.initialize()

        assert await adapter.query(COUNT) == [{"count": 0}]
    finally:
        await adapter.close()
        await relay.stop()
        conftest.drop_graph(graph)
