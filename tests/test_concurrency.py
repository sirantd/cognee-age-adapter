"""Two writers on one graph, in separate pools: no duplicate, no failed write.

On FalkorDB one writer at a time made a write race impossible. On PostgreSQL
two cognify workers (two processes, two pools) can create the same node or the
same edge at the same moment. The unique indexes on the node id and on the edge
identity turn that race into a unique violation, and the adapter runs the
losing write again (homelab#332, stories 19 and 20).

The first test forces the race deterministically: writer A holds its
transaction open after its create step, so writer B's create of the same id
waits on the unique index and then fails. The second test runs many
overlapping writes at once.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from cognee.infrastructure.engine import DataPoint

from cognee_age_adapter.constants import EDGE_LABEL, NODE_LABEL

import conftest

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


class _Ent(DataPoint):
    name: str
    summary: str = ""
    metadata: dict = {"index_fields": ["name"]}


@pytest.fixture
async def second_writer(adapter):
    """A second adapter on the same graph, with its own pool, as a second process."""
    other = conftest.new_adapter(adapter._graph_name)
    await other.initialize()
    try:
        yield other
    finally:
        await other.close()


def _count_statements(adapter) -> list:
    sent = []
    original = adapter._execute

    async def counting(cursor, sql, args):
        sent.append(sql)
        await original(cursor, sql, args)

    adapter._execute = counting
    return sent


async def test_a_lost_create_race_is_retried_not_raised(adapter, second_writer):
    node_id = uuid4()
    created = asyncio.Event()
    original = adapter._execute

    async def hold_after_create(cursor, sql, args):
        await original(cursor, sql, args)
        if "CREATE (:Node {id: wanted})" in sql and not created.is_set():
            created.set()
            # A's insert of node_id is now uncommitted. B's create of the same
            # id waits on the unique index while A sleeps.
            await asyncio.sleep(0.5)

    adapter._execute = hold_after_create
    b_statements = _count_statements(second_writer)

    async def writer_b():
        await created.wait()
        await second_writer.add_nodes([_Ent(id=node_id, name="B", summary="from B")])

    # return_exceptions: both writers finish before an assertion can fail, so a
    # failed B never leaves A's transaction open under the graph teardown.
    results = await asyncio.gather(
        adapter.add_nodes([_Ent(id=node_id, name="A")]), writer_b(), return_exceptions=True
    )
    assert results == [None, None], results

    rows = await adapter.query(
        f"MATCH (n:{NODE_LABEL}) WHERE n.id = $id RETURN n.name AS name, n.summary AS summary",
        {"id": str(node_id)},
    )
    assert rows == [{"name": "B", "summary": "from B"}], "one node, with B's later write"
    # B ran its create, lost, and ran create + update again.
    assert len(b_statements) == 3, b_statements


async def test_overlapping_concurrent_writes_make_each_node_and_edge_once(adapter, second_writer):
    shared = [_Ent(id=uuid4(), name=f"shared {index}") for index in range(20)]
    ids = [str(node.id) for node in shared]
    edges = [(ids[index], ids[index + 1], "next", {}) for index in range(len(ids) - 1)]

    async def writer(instance, order):
        for _ in range(3):
            await instance.add_nodes(shared[::order])
            await instance.add_edges(edges[::order])

    results = await asyncio.gather(
        writer(adapter, 1), writer(second_writer, -1), return_exceptions=True
    )
    assert results == [None, None], results

    counts = await adapter.query(
        f"""
        MATCH (n:{NODE_LABEL}) WITH count(n) AS nodes, count(DISTINCT n.id) AS ids
        MATCH ()-[r:{EDGE_LABEL}]->() RETURN nodes, ids, count(r) AS edges
        """
    )
    assert counts == [{"nodes": 20, "ids": 20, "edges": 19}]
