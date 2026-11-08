"""The fold-vs-attach lost-update race, as an adapter-owned gate.

cognee's contract suite runs
``test_concurrent_folded_writes_and_attaches_keep_every_owner`` for the node
path. ``add_edges`` folds into its write exactly as ``add_nodes`` does, so the
edge test below is coverage that cognee's suite does not give. The node test is
the deliberate overlap: it localizes a regression to this adapter rather than
to a cognee bump.

**The race.** ``add_nodes(..., source_ref_key=...)`` folds the owner key into
the write, one statement. ``attach_node_source_refs`` is a read-then-write pair.
A fold that commits between the attach's read and its write is overwritten, and
the owner it stamped is gone. Two documents of one cognify run that share an
entity are exactly that interleave.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from cognee.infrastructure.databases.provenance import EdgeIdentity, make_source_ref_key
from cognee.infrastructure.engine import DataPoint

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

# Interleavings per writer. The loss is probabilistic per round, and a flaky
# gate for a data-loss bug is not a gate. Measured on the unfixed FalkorDB
# adapter: 6 rounds x 2 writers lost keys on every one of 10 runs.
ROUNDS = 6


class _Ent(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"]}


def _owner_keys(dataset_id, count):
    """``count`` distinct owner keys in one dataset — one per owning data item.

    The race is about *how* owner keys are written, not how finely they are
    scoped, so the coarser builder serves.
    """
    return [make_source_ref_key(dataset_id, uuid4()) for _ in range(count)]


async def test_concurrent_folded_and_attached_node_writes_keep_every_owner(adapter):
    """Two documents folding and attaching against one shared entity lose nothing."""
    shared = _Ent(id=uuid4(), name="Alice")
    node_id = str(shared.id)
    dataset_id = uuid4()

    folded_a, attached_a = _owner_keys(dataset_id, ROUNDS), _owner_keys(dataset_id, ROUNDS)
    folded_b, attached_b = _owner_keys(dataset_id, ROUNDS), _owner_keys(dataset_id, ROUNDS)

    async def writer(folded, attached):
        for fold_key, attach_key in zip(folded, attached):
            await adapter.add_nodes([shared], source_ref_key=fold_key)
            await adapter.attach_node_source_refs([node_id], [attach_key])

    await asyncio.gather(writer(folded_a, attached_a), writer(folded_b, attached_b))

    snapshot = (await adapter.get_node_delete_data([node_id]))[node_id]
    expected = set(folded_a + attached_a + folded_b + attached_b)
    missing = expected - set(snapshot.source_ref_keys)
    assert not missing, f"{len(missing)} of {len(expected)} owner key(s) lost: {sorted(missing)[:3]}"


async def test_concurrent_folded_and_attached_edge_writes_keep_every_owner(adapter):
    """The edge half of the same race — not covered by cognee's own case.

    ``add_edges`` folds into its MERGE exactly as ``add_nodes`` does, and
    ``attach_edge_source_refs`` is the same read-then-write pair, so the edge
    path has the identical lost-update window.
    """
    left, right = _Ent(id=uuid4(), name="Alice"), _Ent(id=uuid4(), name="Rabbit")
    await adapter.add_nodes([left, right])

    edge = (str(left.id), str(right.id), "knows", {"relationship_name": "knows"})
    identity = EdgeIdentity(
        source_id=str(left.id), target_id=str(right.id), relationship_name="knows"
    )
    dataset_id = uuid4()

    folded_a, attached_a = _owner_keys(dataset_id, ROUNDS), _owner_keys(dataset_id, ROUNDS)
    folded_b, attached_b = _owner_keys(dataset_id, ROUNDS), _owner_keys(dataset_id, ROUNDS)

    async def writer(folded, attached):
        for fold_key, attach_key in zip(folded, attached):
            await adapter.add_edges([edge], source_ref_key=fold_key)
            await adapter.attach_edge_source_refs([identity], [attach_key])

    await asyncio.gather(writer(folded_a, attached_a), writer(folded_b, attached_b))

    snapshot = (await adapter.get_edge_delete_data([identity]))[identity]
    expected = set(folded_a + attached_a + folded_b + attached_b)
    missing = expected - set(snapshot.source_ref_keys)
    assert not missing, f"{len(missing)} of {len(expected)} owner key(s) lost: {sorted(missing)[:3]}"
