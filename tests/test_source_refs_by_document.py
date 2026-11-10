"""The per-document source-ref lookups equal the by_dataset-then-filter oracle.

``find_*_source_refs_by_document`` exist so cognee's ``delete_by_document`` can
stop pulling a whole dataset over the wire to find one document's refs. The
homelab core patch swaps them in for exactly this computation, so the only
correct result is the one cognee computes today:

    {k: [ref for ref in refs if parse_source_ref_key(ref).data_id == data_id]
     for k, refs in find_*_source_refs_by_dataset(dataset_id).items()}  # empties dropped

``_oracle_*`` below restates that verbatim, and every case compares against it
rather than against hand-written expectations.

The Cypher pre-filter is ``key CONTAINS $data_id``, a superset. The traps it
must not leak through are built explicitly: the data id as another key's
*dataset* segment, as another key's *chunk* segment, and the same document under
a second dataset on one node.

Ported unchanged from cognee-falkordb-adapter 0.5.4, except the store layout.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from cognee.infrastructure.databases.provenance import (
    EdgeIdentity,
    make_chunk_source_ref_key,
    make_source_ref_key,
    parse_source_ref_key,
)
from cognee.infrastructure.engine import DataPoint

import cognee_age_adapter.adapter as adapter_module
from cognee_age_adapter import NODE_LABEL

import conftest

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


class _Ent(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"]}


class DocumentChunk(DataPoint):
    """Stands in for cognee's chunk: only the type matters to the sweep."""

    text: str
    metadata: dict = {"index_fields": ["text"]}


def _document_refs(refs, data_id):
    """cognee 1.5.4 ``UnifiedStoreEngine.delete_by_document._document_refs``, verbatim."""
    selected = []
    for ref in refs:
        try:
            parsed = parse_source_ref_key(ref)
        except ValueError:
            continue
        if str(parsed.data_id) == str(data_id):
            selected.append(ref)
    return selected


async def _oracle_nodes(adapter, dataset_id, data_id):
    return {
        node_id: document_refs
        for node_id, refs in (await adapter.find_node_source_refs_by_dataset(dataset_id)).items()
        if (document_refs := _document_refs(refs, data_id))
    }


async def _oracle_edges(adapter, dataset_id, data_id):
    return {
        edge: document_refs
        for edge, refs in (await adapter.find_edge_source_refs_by_dataset(dataset_id)).items()
        if (document_refs := _document_refs(refs, data_id))
    }


async def _assert_matches_oracle(adapter, dataset_id, data_id):
    ds, doc = str(dataset_id), str(data_id)
    nodes = await adapter.find_node_source_refs_by_document(ds, doc)
    edges = await adapter.find_edge_source_refs_by_document(ds, doc)
    assert nodes == await _oracle_nodes(adapter, ds, doc)
    assert edges == await _oracle_edges(adapter, ds, doc)
    return nodes, edges


def _edge(left, right, name):
    return (
        (str(left.id), str(right.id), name, {"relationship_name": name}),
        EdgeIdentity(source_id=str(left.id), target_id=str(right.id), relationship_name=name),
    )


@pytest.fixture
async def world(adapter):
    """Two datasets, three documents, v1 and v2 keys, shared nodes and edges.

    ``doc_a`` belongs to BOTH datasets — one data item added twice — so its keys
    under ``ds_2`` sit on the same shared node as its keys under ``ds_1``.
    """
    ds_1, ds_2 = uuid4(), uuid4()
    doc_a, doc_b, doc_c = uuid4(), uuid4(), uuid4()

    alice, rabbit, queen, hatter = (
        _Ent(id=uuid4(), name=name) for name in ("Alice", "Rabbit", "Queen", "Hatter")
    )

    # v1 folded writes: alice + rabbit from doc_a, queen from doc_b, hatter from doc_c.
    await adapter.add_nodes([alice, rabbit], source_ref_key=make_source_ref_key(ds_1, doc_a))
    await adapter.add_nodes([queen], source_ref_key=make_source_ref_key(ds_1, doc_b))
    await adapter.add_nodes([hatter], source_ref_key=make_source_ref_key(ds_2, doc_c))

    # Shared: alice is also owned by doc_b (v1 + v2), doc_c, and doc_a under ds_2.
    await adapter.attach_node_source_refs(
        [str(alice.id)],
        [
            make_source_ref_key(ds_1, doc_b),
            make_chunk_source_ref_key(ds_1, doc_b, uuid4()),
            make_chunk_source_ref_key(ds_1, doc_a, uuid4()),
            make_chunk_source_ref_key(ds_1, doc_a, uuid4()),
            make_source_ref_key(ds_2, doc_c),
            make_source_ref_key(ds_2, doc_a),
            make_chunk_source_ref_key(ds_2, doc_a, uuid4()),
        ],
    )
    # v2-only ownership: queen gains a doc_a chunk key and nothing v1 from doc_a.
    await adapter.attach_node_source_refs(
        [str(queen.id)], [make_chunk_source_ref_key(ds_1, doc_a, uuid4())]
    )

    knows, knows_id = _edge(alice, rabbit, "knows")
    fears, fears_id = _edge(alice, queen, "fears")
    hosts, hosts_id = _edge(hatter, alice, "hosts")
    await adapter.add_edges([knows], source_ref_key=make_source_ref_key(ds_1, doc_a))
    await adapter.add_edges([fears], source_ref_key=make_source_ref_key(ds_1, doc_b))
    await adapter.add_edges([hosts], source_ref_key=make_source_ref_key(ds_2, doc_c))
    await adapter.attach_edge_source_refs(
        [knows_id],
        [
            make_chunk_source_ref_key(ds_1, doc_a, uuid4()),
            make_chunk_source_ref_key(ds_1, doc_b, uuid4()),
            make_source_ref_key(ds_2, doc_a),
        ],
    )
    await adapter.attach_edge_source_refs(
        [fears_id], [make_chunk_source_ref_key(ds_1, doc_a, uuid4())]
    )

    return {
        "datasets": (ds_1, ds_2),
        "documents": (doc_a, doc_b, doc_c),
        "nodes": {"alice": alice, "rabbit": rabbit, "queen": queen, "hatter": hatter},
        "edges": {"knows": knows_id, "fears": fears_id, "hosts": hosts_id},
    }


async def test_every_dataset_document_pair_matches_the_oracle(adapter, world):
    """All 2 x 3 (dataset, document) pairs — owning and non-owning — equal the oracle."""
    for dataset_id in world["datasets"]:
        for data_id in world["documents"]:
            await _assert_matches_oracle(adapter, dataset_id, data_id)


async def test_anchor_slices_and_no_dataset_index_condition(adapter, world, monkeypatch):
    """🚨 homelab#332 D3. With ``r.source_dataset_ids @> [$dataset_id]`` in the
    anchored read, the planner started from the GIN index and read every edge
    of the dataset: 138 s for a document of 22 nodes on production. The
    anchored read checks the dataset after the anchor, and goes in slices of
    ANCHOR_BATCH anchors. Slices of one anchor must give the oracle's result."""
    monkeypatch.setattr(adapter_module, "ANCHOR_BATCH", 1)
    sent = conftest.record_sql(adapter)
    try:
        for dataset_id in world["datasets"]:
            for data_id in world["documents"]:
                await _assert_matches_oracle(adapter, dataset_id, data_id)
    finally:
        del adapter._execute
    anchored = [sql for sql, args in sent if "$anchor_ids" in sql]
    assert len(anchored) > len(world["datasets"]) * len(world["documents"]), "one slice per anchor"
    assert not [sql for sql in anchored if "source_dataset_ids @>" in sql], anchored[0]


async def test_shared_artifacts_return_only_this_documents_refs(adapter, world):
    """A shared node/edge carries several owners; only this document's keys come back."""
    ds_1, ds_2 = world["datasets"]
    doc_a, _doc_b, _doc_c = world["documents"]
    alice = str(world["nodes"]["alice"].id)
    queen = str(world["nodes"]["queen"].id)

    nodes, edges = await _assert_matches_oracle(adapter, ds_1, doc_a)

    # alice: the v1 fold + two v2 chunk keys of doc_a under ds_1 — nothing of
    # doc_b, doc_c, or doc_a's own keys under ds_2.
    assert len(nodes[alice]) == 3
    assert all(
        str(p.dataset_id) == str(ds_1) and p.data_id == doc_a
        for p in map(parse_source_ref_key, nodes[alice])
    )
    assert {parse_source_ref_key(k).version for k in nodes[alice]} == {1, 2}
    # queen: owned by doc_a through a v2 key only.
    assert [parse_source_ref_key(k).version for k in nodes[queen]] == [2]

    assert set(edges) == {world["edges"]["knows"], world["edges"]["fears"]}
    assert len(edges[world["edges"]["knows"]]) == 2  # v1 fold + v2 chunk, not ds_2's

    # The same document under the other dataset resolves to that dataset's keys.
    nodes_2, edges_2 = await _assert_matches_oracle(adapter, ds_2, doc_a)
    assert set(nodes_2) == {alice}
    assert len(nodes_2[alice]) == 2
    assert set(edges_2) == {world["edges"]["knows"]}


async def test_document_with_no_refs_is_empty(adapter, world):
    ds_1, _ds_2 = world["datasets"]
    stranger = str(uuid4())
    assert await adapter.find_node_source_refs_by_document(str(ds_1), stranger) == {}
    assert await adapter.find_edge_source_refs_by_document(str(ds_1), stranger) == {}
    await _assert_matches_oracle(adapter, ds_1, stranger)


async def test_data_id_in_another_segment_does_not_match(adapter, world):
    """``CONTAINS`` matches the id anywhere; the exact filter must reject non-data segments.

    The probe id appears as a key's DATASET segment and as a v2 key's CHUNK
    segment — both contain it literally, neither is owned by it as a document.
    """
    ds_1, _ds_2 = world["datasets"]
    doc_a, doc_b, _doc_c = world["documents"]
    probe = uuid4()
    rabbit = str(world["nodes"]["rabbit"].id)

    await adapter.attach_node_source_refs(
        [rabbit],
        [
            make_source_ref_key(probe, doc_b),  # probe as dataset segment
            make_chunk_source_ref_key(ds_1, doc_b, probe),  # probe as chunk segment
        ],
    )
    await adapter.attach_edge_source_refs(
        [world["edges"]["fears"]], [make_chunk_source_ref_key(ds_1, doc_a, probe)]
    )

    assert await adapter.find_node_source_refs_by_document(str(ds_1), str(probe)) == {}
    assert await adapter.find_edge_source_refs_by_document(str(ds_1), str(probe)) == {}
    await _assert_matches_oracle(adapter, ds_1, probe)
    # The chunk-segment keys still count for the documents that DO own them.
    for doc in (doc_a, doc_b):
        await _assert_matches_oracle(adapter, ds_1, doc)


async def test_partial_data_id_does_not_match(adapter, world):
    """A prefix or substring of a real data id is not that document."""
    ds_1, _ds_2 = world["datasets"]
    doc_a = str(world["documents"][0])
    for partial in (doc_a[:8], doc_a[:-1], doc_a[9:]):
        assert await adapter.find_node_source_refs_by_document(str(ds_1), partial) == {}
        assert await adapter.find_edge_source_refs_by_document(str(ds_1), partial) == {}
        await _assert_matches_oracle(adapter, ds_1, partial)
    assert await adapter.find_node_source_refs_by_document(str(ds_1), "") == {}
    assert await adapter.find_edge_source_refs_by_document(str(ds_1), "") == {}


async def test_undecomposable_key_is_skipped_not_raised(adapter, world):
    """A malformed key containing the data id is dropped, as delete_by_document drops it.

    Written raw: the adapter's own attach path derives dataset ids from every
    key and would refuse it. No oracle comparison here — the by_dataset lookup
    raises on such a key rather than skipping it.
    """
    ds_1, _ds_2 = world["datasets"]
    doc_a = world["documents"][0]
    alice = str(world["nodes"]["alice"].id)
    before = await adapter.find_node_source_refs_by_document(str(ds_1), str(doc_a))

    await adapter.query(
        f"""
        MATCH (n:{NODE_LABEL}) WHERE n.id = $id
        SET n.source_ref_keys = n.source_ref_keys + [$bad]
        """,
        {"id": alice, "bad": f"source_ref:v9:{ds_1}:{doc_a}"},
    )

    assert await adapter.find_node_source_refs_by_document(str(ds_1), str(doc_a)) == before


# --- the edge lookup's invariant ------------------------------------------------
#
# Edges are found by anchoring on the document's own nodes, plus a sweep of
# chunk->chunk edges (see ``find_edge_source_refs_by_document``). The three
# cases below pin what that covers and what it does not.


async def test_doc_specific_edge_between_nodes_of_different_documents(adapter, world):
    """A shared entity + an edge only one document states, to another document's node.

    ``rabbit`` is shared by doc_a and doc_b; ``hatter`` belongs to doc_c alone.
    The edge carries doc_b's ref only: it is found through ``rabbit``, and not
    returned for doc_c although one endpoint is doc_c's.
    """
    ds_1, ds_2 = world["datasets"]
    doc_a, doc_b, doc_c = world["documents"]
    rabbit, hatter = world["nodes"]["rabbit"], world["nodes"]["hatter"]

    await adapter.attach_node_source_refs(
        [str(rabbit.id)], [make_chunk_source_ref_key(ds_1, doc_b, uuid4())]
    )
    chases, chases_id = _edge(rabbit, hatter, "chases")
    await adapter.add_edges([chases], source_ref_key=make_chunk_source_ref_key(ds_1, doc_b, uuid4()))

    _nodes, edges = await _assert_matches_oracle(adapter, ds_1, doc_b)
    assert chases_id in edges
    for dataset_id in (ds_1, ds_2):
        for data_id in (doc_a, doc_c):
            _nodes, edges = await _assert_matches_oracle(adapter, dataset_id, data_id)
            assert chases_id not in edges


async def test_chunk_association_edge_between_other_documents_chunks_is_found(adapter, world):
    """The ``create_chunk_associations`` case — the reason the chunk sweep exists.

    Both chunks belong to doc_c; the association edge between them carries
    doc_a's ref (its endpoints were resolved by a collection-wide vector search).
    No node doc_a owns touches it, so only the sweep can find it.
    """
    ds_1, _ds_2 = world["datasets"]
    doc_a, _doc_b, doc_c = world["documents"]
    left, right = DocumentChunk(id=uuid4(), text="left"), DocumentChunk(id=uuid4(), text="right")
    await adapter.add_nodes([left, right], source_ref_key=make_source_ref_key(ds_1, doc_c))
    association, association_id = _edge(left, right, "is_similar_to")
    await adapter.add_edges([association], source_ref_key=make_source_ref_key(ds_1, doc_a))

    doc_a_nodes = await adapter.find_node_source_refs_by_document(str(ds_1), str(doc_a))
    assert str(left.id) not in doc_a_nodes and str(right.id) not in doc_a_nodes

    _nodes, edges = await _assert_matches_oracle(adapter, ds_1, doc_a)
    assert edges[association_id] == [make_source_ref_key(ds_1, doc_a)]
    _nodes, edges = await _assert_matches_oracle(adapter, ds_1, doc_c)
    assert association_id not in edges


async def test_known_limitation_unanchored_non_chunk_edge_is_missed(adapter, world):
    """🚨 Pins the one shape the lookup does NOT cover — and that it is the only one.

    An edge carrying doc_a's ref, whose endpoints do not own doc_a and are not
    both chunks. No cognee 1.5.4 write path produces this (the invariant is
    listed in ``find_edge_source_refs_by_document``). If a cognee bump adds one,
    this is the shape that leaks: the edge keeps a stale ref, nothing is
    over-deleted. When this test's premise changes, re-verify the invariant.
    """
    ds_1, _ds_2 = world["datasets"]
    doc_a, _doc_b, doc_c = world["documents"]
    left, right = _Ent(id=uuid4(), name="Dodo"), _Ent(id=uuid4(), name="Gryphon")
    await adapter.add_nodes([left, right], source_ref_key=make_source_ref_key(ds_1, doc_c))
    stray, stray_id = _edge(left, right, "argues_with")
    await adapter.add_edges([stray], source_ref_key=make_source_ref_key(ds_1, doc_a))

    edges = await adapter.find_edge_source_refs_by_document(str(ds_1), str(doc_a))
    oracle = await _oracle_edges(adapter, str(ds_1), str(doc_a))
    assert stray_id in oracle and stray_id not in edges
    assert {k: v for k, v in oracle.items() if k != stray_id} == edges
