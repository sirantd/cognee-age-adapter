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

from cognee.infrastructure.databases.provenance import make_source_ref_key
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
    b_statements = conftest.record_sql(second_writer)

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


def _hold_after(adapter, marker: str, held: asyncio.Event) -> None:
    """Make ``adapter`` keep its transaction open for 0.5 s after the first
    statement that contains ``marker``, so a second writer runs into it."""
    original = adapter._execute

    async def holding(cursor, sql, args):
        await original(cursor, sql, args)
        if marker in sql and not held.is_set():
            held.set()
            await asyncio.sleep(0.5)

    adapter._execute = holding


async def test_a_lost_update_race_is_retried_not_raised(adapter, second_writer):
    """🚨 AGE 1.8.0 fails a SET on a row that a concurrent transaction updated
    (``Entity failed to be updated: 3``) instead of re-checking the row. Writer
    B's update waits on A's row, fails, and runs again."""
    node_id = uuid4()
    await adapter.add_nodes([_Ent(id=node_id, name="start")])
    held = asyncio.Event()
    _hold_after(adapter, "SET n += props", held)
    b_statements = conftest.record_sql(second_writer)

    async def writer_b():
        await held.wait()
        await second_writer.add_nodes([_Ent(id=node_id, name="B")])

    results = await asyncio.gather(
        adapter.add_nodes([_Ent(id=node_id, name="A")]), writer_b(), return_exceptions=True
    )
    assert results == [None, None], results

    assert (await adapter.get_node(str(node_id)))["name"] == "B"
    assert len(b_statements) == 4, "B ran create + update, lost, and ran both again"


async def test_attaches_from_two_processes_keep_both_owner_keys(adapter, second_writer):
    """The provenance write is a read-modify-write. When B's write loses to A's,
    B reads again, so A's key is not overwritten."""
    node = _Ent(id=uuid4(), name="shared")
    await adapter.add_nodes([node])
    dataset_id = uuid4()
    key_a, key_b = (make_source_ref_key(dataset_id, uuid4()) for _ in range(2))
    held = asyncio.Event()
    _hold_after(adapter, "SET n.source_ref_keys = row.refs", held)

    async def writer_b():
        await held.wait()
        await second_writer.attach_node_source_refs([str(node.id)], [key_b])

    results = await asyncio.gather(
        adapter.attach_node_source_refs([str(node.id)], [key_a]),
        writer_b(),
        return_exceptions=True,
    )
    assert results == [None, None], results

    snapshot = (await adapter.get_node_delete_data([str(node.id)]))[str(node.id)]
    assert set(snapshot.source_ref_keys) == {key_a, key_b}


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
