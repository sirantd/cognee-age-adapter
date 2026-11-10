"""Writers on one graph, in separate pools: no duplicate, no failed write.

On FalkorDB one writer at a time made a write race impossible. On PostgreSQL
cognify workers in several processes (several pools) can write the same node or
the same edge at the same moment (homelab#332, stories 19 and 20). Each node
and edge write first takes an advisory lock for each node id or edge identity,
so the second writer waits for the first to commit (D4). The unique indexes
stay, and a write that still loses a race runs again.

The first tests force the overlap deterministically: writer A holds its
transaction open, and writer B's write of the same node waits. The later tests
run many overlapping writers at once, as cognify writes.
"""

from __future__ import annotations

import asyncio
import random
from collections import Counter
from uuid import uuid4

import pytest

from cognee.infrastructure.databases.provenance import EdgeIdentity, make_source_ref_key
from cognee.infrastructure.engine import DataPoint

import cognee_age_adapter.adapter as adapter_module
from cognee_age_adapter.constants import EDGE_LABEL, LOCK_BATCH, NODE_LABEL

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


def _locks(statements) -> int:
    return sum("pg_advisory_xact_lock" in sql for sql, _ in statements)


async def test_a_concurrent_create_of_one_id_waits_and_is_not_retried(adapter, second_writer):
    node_id = uuid4()
    created = asyncio.Event()
    original = adapter._execute

    async def hold_after_create(cursor, sql, args):
        await original(cursor, sql, args)
        if "CREATE (:Node {id: wanted})" in sql and not created.is_set():
            created.set()
            # A's insert of node_id is now uncommitted. B's lock of the same
            # id waits for A's commit while A sleeps.
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
    # B waited in its lock, then ran create (which found A's node) and update
    # once. Before D4, B ran its create, lost the unique index, and ran again.
    assert len(b_statements) == 3 and _locks(b_statements) == 1, b_statements


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


async def test_a_concurrent_update_of_one_node_waits_and_is_not_retried(adapter, second_writer):
    """🚨 AGE 1.8.0 fails a SET on a row that a concurrent transaction updated
    (``Entity failed to be updated: 3``) instead of re-checking the row. So
    writer B waits in its lock until A commits, and its update then reads the
    committed row. Before D4, B's update failed and ran again."""
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
    assert len(b_statements) == 3 and _locks(b_statements) == 1, b_statements


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


# The cognify-shaped load of homelab#332 D4: writers on one graph, and shared
# nodes and edges that several writers write in one round.
WRITERS = 6
ROUNDS = 8
RING = 40


async def test_many_cognify_shaped_writers_neither_fail_nor_retry(adapter, monkeypatch):
    """🚨 homelab#332 D4. Measured on production on 2026-11-10 at adapter
    ``aa577fa``: with 5 writers, 9 of 80 batches failed after WRITE_ATTEMPTS
    with ``Entity failed to be updated: 3``, and their nodes and edges were
    lost. Each writer here is an adapter with its own pool, as a process is.
    Each round writes as cognify does (``add_data_points``): a folded
    ``add_nodes`` and ``add_edges``, then an attach of a chunk key to the
    shared nodes and edges. With the locks, no write waits for a retry, and
    every owner key stays."""
    retries: Counter = Counter()
    original = adapter_module._wait_before_retry

    async def counting(attempt, exc):
        retries[type(exc).__name__] += 1
        await original(attempt, exc)

    monkeypatch.setattr(adapter_module, "_wait_before_retry", counting)

    dataset_id = uuid4()
    shared = sorted(str(uuid4()) for _ in range(RING))
    ring = [(shared[index], shared[(index + 1) % RING], "next") for index in range(RING)]
    expected_keys: dict = {}

    def expect(artifact, key):
        expected_keys.setdefault(artifact, set()).add(key)

    def rounds_of(writer: int):
        rng = random.Random(writer)
        rounds = []
        for _ in range(ROUNDS):
            fold = make_source_ref_key(dataset_id, uuid4())
            extra = make_source_ref_key(dataset_id, uuid4())
            ring_edges = rng.sample(ring, 10)
            ends = sorted({end for s, t, _ in ring_edges for end in (s, t)})
            own = [str(uuid4()) for _ in range(20)]
            edges = [(node, rng.choice(ends), "mentions", {}) for node in own]
            edges += [(s, t, rel, {"edge_text": rel}) for s, t, rel in ring_edges]
            for node in [*own, *ends]:
                expect(node, fold)
            for node in ends:
                expect(node, extra)
            for s, t, rel, _ in edges:
                expect((s, t, rel), fold)
            for edge in ring_edges:
                expect(edge, extra)
            rounds.append((fold, extra, own, ends, edges, ring_edges))
        return rounds

    async def write(instance, rounds):
        for fold, extra, own, ends, edges, ring_edges in rounds:
            nodes = [_Ent(id=node, name=node[:8]) for node in [*own, *ends]]
            await instance.add_nodes(nodes, source_ref_key=fold)
            await instance.attach_node_source_refs(ends, [extra])
            await instance.add_edges(edges, source_ref_key=fold)
            await instance.attach_edge_source_refs([EdgeIdentity(*edge) for edge in ring_edges], [extra])

    others = [conftest.new_adapter(adapter._graph_name) for _ in range(WRITERS - 1)]
    try:
        writers = [adapter, *others]
        plans = [rounds_of(index) for index in range(WRITERS)]
        results = await asyncio.gather(
            *(write(instance, plan) for instance, plan in zip(writers, plans)),
            return_exceptions=True,
        )
    finally:
        for other in others:
            await other.close()
    assert results == [None] * WRITERS, results
    assert retries == Counter(), f"writes lost a race and ran again: {dict(retries)}"

    nodes = [artifact for artifact in expected_keys if isinstance(artifact, str)]
    edges = [artifact for artifact in expected_keys if isinstance(artifact, tuple)]
    stored_nodes = await adapter.get_node_delete_data(nodes)
    stored_edges = await adapter.get_edge_delete_data([EdgeIdentity(*edge) for edge in edges])
    assert len(stored_nodes) == len(nodes) and len(stored_edges) == len(edges)
    lost = [
        node for node in nodes if set(stored_nodes[node].source_ref_keys) != expected_keys[node]
    ] + [
        edge
        for edge in edges
        if set(stored_edges[EdgeIdentity(*edge)].source_ref_keys) != expected_keys[edge]
    ]
    assert not lost, f"{len(lost)} nodes and edges lost an owner key"


LARGE_WRITERS = 5
LARGE_BATCH = 4000


async def test_concurrent_large_batches_do_not_fill_the_lock_table(adapter):
    """🚨 The advisory locks of every session share one lock table. On a
    default server (max_locks_per_transaction 64, max_connections 100), 14,000
    locks fit and 16,000 fail with ``out of shared memory`` (measured). Five
    writers of 4,000 nodes at once, each in one transaction, would hold
    20,000. So a write locks at most LOCK_BATCH nodes in one transaction."""
    others = [conftest.new_adapter(adapter._graph_name) for _ in range(LARGE_WRITERS - 1)]
    sent = conftest.record_sql(adapter)
    try:
        batches = [
            [_Ent(id=uuid4(), name=f"{w} {i}") for i in range(LARGE_BATCH)]
            for w in range(LARGE_WRITERS)
        ]
        results = await asyncio.gather(
            *(writer.add_nodes(batch) for writer, batch in zip([adapter, *others], batches)),
            return_exceptions=True,
        )
    finally:
        del adapter._execute
        for other in others:
            await other.close()
    assert results == [None] * LARGE_WRITERS, results
    locked = [len(args[0]) for sql, args in sent if "pg_advisory_xact_lock" in sql]
    assert locked and max(locked) <= LOCK_BATCH, locked

    rows = await adapter.query(f"MATCH (n:{NODE_LABEL}) RETURN count(n) AS nodes")
    assert rows == [{"nodes": LARGE_WRITERS * LARGE_BATCH}]
