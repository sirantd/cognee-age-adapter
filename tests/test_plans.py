"""The plan gate: every query the adapter emits reads its label tables through an index.

homelab#332 (story 24): a missing index must fail the build, not the 03:00
drain. A test of "the index exists" is not enough, because the plan is what
silently changes. So this file runs a workload that calls EVERY public method,
records every statement at the one place the adapter sends SQL (``_execute``),
and asks PostgreSQL for the plan of exactly that statement with its real
parameters.

The rules for a plan, on small test data:

* ``enable_seqscan``, ``enable_hashjoin`` and ``enable_mergejoin`` are off.
  PostgreSQL can choose a sequential scan or a join over a whole table on a
  small table even when the index is correct. With the three off, the planner
  takes an index whenever one serves the query.
* ⚠ So a sequential scan is no longer the only sign of a missing index: with
  seqscan off, the planner reads a whole table through an unrelated index
  instead. The check is therefore: (a) no ``Disabled: true`` in the plan
  (PostgreSQL 18 marks a node it had to take although it was disabled), and
  (b) every scan of a label table carries an ``Index Cond`` or a
  ``Recheck Cond``.

``WHOLE_GRAPH_READS`` is the closed list of methods whose job is to read the
whole graph. They are in the workload, so the census sees them, but their
plans are not checked.
"""

from __future__ import annotations

import inspect
import re
from uuid import uuid4

import pytest

from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
from cognee.infrastructure.databases.provenance import (
    EdgeIdentity,
    make_chunk_source_ref_key,
    make_source_ref_key,
)
from cognee.infrastructure.engine import DataPoint

from cognee_age_adapter import AGEAdapter
from cognee_age_adapter.constants import EDGE_LABEL, METADATA_LABEL, NODE_LABEL

import conftest

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

LABEL_TABLES = (NODE_LABEL, EDGE_LABEL, METADATA_LABEL, "_ag_label_vertex", "_ag_label_edge")

# Methods whose job is a read or a delete of the whole graph. Asserted by name.
WHOLE_GRAPH_READS = frozenset(
    {"delete_graph", "get_graph_data", "get_graph_metrics", "get_triplets_batch", "is_empty"}
)

# Public methods that send no data query: DDL, the pool, a pure function, and
# the generic Cypher path, whose queries come from the caller.
NOT_A_DATA_QUERY = frozenset({"close", "initialize", "query", "serialize_properties"})

# Nodes and edges of filler in the seeded graph; see ``seeded``.
FILLER = 3000


class Entity(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"]}


class DocumentChunk(DataPoint):
    text: str
    document_id: str = ""
    metadata: dict = {"index_fields": ["text"]}


class NodeSet(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"]}


def _workload(seed):
    """``{method: call}`` for every public data method, on the seeded graph."""
    a, b, c, chunk_1, chunk_2, nodeset = seed["ids"]
    dataset_id, data_id, key = seed["dataset_id"], seed["data_id"], seed["key"]
    edge = EdgeIdentity(source_id=a, target_id=b, relationship_name="knows")
    run_id = seed["run_id"]
    return {
        "has_node": lambda x: x.has_node(a),
        "is_empty": lambda x: x.is_empty(),
        "add_node": lambda x: x.add_node(Entity(id=uuid4(), name="new")),
        "add_nodes": lambda x: x.add_nodes(
            [Entity(id=uuid4(), name="n1"), Entity(id=uuid4(), name="n2")],
            source_ref_key=key,
            pipeline_run_id=run_id,
        ),
        "delete_node": lambda x: x.delete_node(str(uuid4())),
        "delete_nodes": lambda x: x.delete_nodes([str(uuid4()), str(uuid4())]),
        "remove_belongs_to_set_tags": _both(
            lambda x: x.remove_belongs_to_set_tags(["Dev"]),
            lambda x: x.remove_belongs_to_set_tags(["Dev"], node_ids=[a]),
        ),
        "get_node": lambda x: x.get_node(a),
        "get_nodes": lambda x: x.get_nodes([a, b]),
        "get_neighbors": lambda x: x.get_neighbors(a),
        "get_connections": lambda x: x.get_connections(b),
        "add_edge": lambda x: x.add_edge(a, c, "likes", {"edge_text": "t"}),
        "add_edges": lambda x: x.add_edges(
            [(b, c, "trusts", {"edge_text": "t"})], source_ref_key=key, pipeline_run_id=run_id
        ),
        "has_edge": lambda x: x.has_edge(a, b, "knows"),
        "has_edges": lambda x: x.has_edges([(a, b, "knows", {})]),
        "get_edges": lambda x: x.get_edges(a),
        "attach_node_source_refs": lambda x: x.attach_node_source_refs([a], [key], run_id),
        "attach_edge_source_refs": lambda x: x.attach_edge_source_refs([edge], [key], run_id),
        "remove_node_source_refs": lambda x: x.remove_node_source_refs([c], [key]),
        "remove_edge_source_refs": lambda x: x.remove_edge_source_refs([edge], ["absent"]),
        "delete_edge_triples": lambda x: x.delete_edge_triples(
            [EdgeIdentity(source_id=c, target_id=a, relationship_name="absent")]
        ),
        "get_node_delete_data": lambda x: x.get_node_delete_data([a, b]),
        "get_edge_delete_data": lambda x: x.get_edge_delete_data([edge]),
        "find_nodes_by_source_ref": lambda x: x.find_nodes_by_source_ref(key),
        "find_edges_by_source_ref": lambda x: x.find_edges_by_source_ref(key),
        "find_node_source_refs_by_dataset": lambda x: x.find_node_source_refs_by_dataset(dataset_id),
        "find_edge_source_refs_by_dataset": lambda x: x.find_edge_source_refs_by_dataset(dataset_id),
        "find_node_source_refs_by_document": lambda x: x.find_node_source_refs_by_document(
            dataset_id, data_id
        ),
        "find_edge_source_refs_by_document": lambda x: x.find_edge_source_refs_by_document(
            dataset_id, data_id
        ),
        "find_node_source_refs_by_pipeline_run": lambda x: x.find_node_source_refs_by_pipeline_run(
            run_id
        ),
        "find_edge_source_refs_by_pipeline_run": lambda x: x.find_edge_source_refs_by_pipeline_run(
            run_id
        ),
        "set_graph_metadata": lambda x: x.set_graph_metadata({"k": "v"}),
        "get_graph_metadata": lambda x: x.get_graph_metadata(),
        "get_node_truth_state": lambda x: x.get_node_truth_state([a]),
        "set_node_truth_state": lambda x: x.set_node_truth_state(
            {a: {"truth_alignment": ["s"], "truth_epoch": 1}, b: {"truth_alignment": []}}
        ),
        "get_id_filtered_graph_data": lambda x: x.get_id_filtered_graph_data([a]),
        "get_neighborhood": _both(
            lambda x: x.get_neighborhood([a], depth=2),
            lambda x: x.get_neighborhood([a], depth=2, edge_types=["knows"]),
        ),
        "get_nodeset_subgraph": _both(
            lambda x: x.get_nodeset_subgraph(NodeSet, ["Dev"]),
            lambda x: x.get_nodeset_subgraph(NodeSet, ["Dev"], node_name_filter_operator="AND"),
        ),
        # cognee filters on ``type`` and ``id`` (see the method's docstring).
        "get_filtered_graph_data": _both(
            lambda x: x.get_filtered_graph_data([{"type": ["Entity"]}]),
            lambda x: x.get_filtered_graph_data([{"id": [a, b], "type": ["Entity"]}]),
        ),
        "get_graph_data": lambda x: x.get_graph_data(),
        "get_graph_metrics": lambda x: x.get_graph_metrics(include_optional=True),
        "get_triplets_batch": lambda x: x.get_triplets_batch(0, 10),
        "delete_graph": lambda x: x.delete_graph(),
    }


def _both(first, second):
    async def call(adapter):
        await first(adapter)
        await second(adapter)

    return call


def _public_data_methods() -> frozenset[str]:
    own = set()
    for klass in AGEAdapter.__mro__:
        if klass is GraphDBInterface:
            break
        own |= {
            name
            for name, value in vars(klass).items()
            if not name.startswith("_") and inspect.isfunction(value)
        }
    return frozenset(own - NOT_A_DATA_QUERY)


@pytest.fixture
async def seeded(adapter):
    """A small graph: three entities, two chunks of one document, one NodeSet."""
    dataset_id, data_id = uuid4(), uuid4()
    entities = [Entity(id=uuid4(), name=name, belongs_to_set=["Dev"]) for name in "abc"]
    chunks = [DocumentChunk(id=uuid4(), text=f"t{i}", document_id=str(data_id)) for i in (1, 2)]
    nodeset = NodeSet(id=uuid4(), name="Dev")
    key = make_source_ref_key(dataset_id, data_id)
    chunk_key = make_chunk_source_ref_key(dataset_id, data_id, chunks[0].id)
    run_id = str(uuid4())
    await adapter.add_nodes([*entities, *chunks, nodeset], source_ref_key=key, pipeline_run_id=run_id)
    ids = [str(node.id) for node in (*entities, *chunks, nodeset)]
    a, b, c, chunk_1, chunk_2, ns = ids
    await adapter.add_edges(
        [
            (a, b, "knows", {"edge_text": "ab"}),
            (b, c, "likes", {"edge_text": "bc"}),
            (chunk_1, chunk_2, "next_chunk", {}),
            (a, ns, "belongs_to_set", {}),
        ],
        source_ref_key=chunk_key,
        pipeline_run_id=run_id,
    )
    # 🚨 Filler, so that a full read of a label costs more than an index
    # lookup. Measured: on 4 edges the planner read the whole edge table
    # through an unrelated index although the ``start_id`` index served the
    # query; the cost of 4 rows is below the cost of any lookup.
    await adapter.query(
        f"""
        UNWIND range(1, $count) AS i
        CREATE (:{NODE_LABEL} {{id: 'filler-' + toString(i), type: 'Filler', name: toString(i),
                               belongs_to_set: ['filler'], source_ref_keys: ['filler'],
                               source_dataset_ids: ['filler'], source_run_ids: ['filler']}})
        """,
        {"count": FILLER},
    )
    await adapter.query(
        f"""
        UNWIND range(1, $count - 1) AS i
        MATCH (s:{NODE_LABEL}) WHERE s.id = 'filler-' + toString(i)
        MATCH (t:{NODE_LABEL}) WHERE t.id = 'filler-' + toString(i + 1)
        CREATE (s)-[:{EDGE_LABEL} {{relationship_name: 'filler', source_ref_keys: ['filler'],
                                   source_dataset_ids: ['filler'], source_run_ids: ['filler']}}]->(t)
        """,
        {"count": FILLER},
    )
    with conftest.connect() as conn:
        for label in (NODE_LABEL, EDGE_LABEL):
            conn.execute(f'ANALYZE "{adapter._graph_name}"."{label}"')
    return {
        "ids": ids,
        "dataset_id": str(dataset_id),
        "data_id": str(data_id),
        "key": key,
        "run_id": run_id,
    }


async def _record(adapter, call):
    """Run a real adapter call; return every (sql, args) it sent."""
    sent = []
    original = adapter._execute

    async def recording(cursor, sql, args):
        sent.append((sql, args))
        await original(cursor, sql, args)

    adapter._execute = recording
    try:
        await call(adapter)
    finally:
        del adapter._execute
    return sent


def _plan(sql: str, args) -> str:
    with conftest.connect() as conn:
        for knob in ("enable_seqscan", "enable_hashjoin", "enable_mergejoin"):
            conn.execute(f"SET {knob} = off")
        return "\n".join(row[0] for row in conn.execute("EXPLAIN " + sql, args).fetchall())


_SCAN = re.compile(
    r"(?P<kind>Seq Scan|Index Scan|Index Only Scan|Bitmap Heap Scan)"
    r"(?: using \S+)? on (?:\S+\.)?\"?(?P<table>[A-Za-z_]+)\"?"
)


def _unindexed_label_scans(plan: str) -> list[str]:
    """Scans of a label table that are not driven by an index condition."""
    lines = plan.splitlines()
    problems = []
    for index, line in enumerate(lines):
        match = _SCAN.search(line)
        if not match or match.group("table") not in LABEL_TABLES:
            continue
        details = []
        for follower in lines[index + 1 :]:
            if "->" in follower or _SCAN.search(follower):
                break
            details.append(follower)
        has_condition = any("Index Cond:" in d or "Recheck Cond:" in d for d in details)
        if match.group("kind") == "Seq Scan" or not has_condition:
            problems.append(line.strip())
    return problems


def test_the_workload_calls_every_public_data_method():
    """🚨 A new method fails here until the workload calls it, so its plan is checked."""
    seed = {"ids": ["x"] * 6, "dataset_id": "d", "data_id": "d", "key": "k", "run_id": "r"}
    covered = frozenset(_workload(seed))
    required = _public_data_methods()
    assert covered == required, (
        f"not in the plan workload: {sorted(required - covered)}; "
        f"not a public data method: {sorted(covered - required)}"
    )
    assert WHOLE_GRAPH_READS <= covered


@pytest.mark.parametrize(
    "method",
    sorted(_workload({"ids": ["x"] * 6, "dataset_id": "", "data_id": "", "key": "", "run_id": ""})),
)
async def test_every_emitted_query_reads_its_labels_through_an_index(adapter, seeded, method):
    sent = await _record(adapter, _workload(seeded)[method])
    cypher = [(sql, args) for sql, args in sent if sql.startswith("SELECT * FROM ag_catalog.cypher")]
    assert cypher, f"{method} sent no Cypher statement"
    if method in WHOLE_GRAPH_READS:
        return

    for sql, args in cypher:
        plan = _plan(sql, args)
        assert "Disabled: true" not in plan, f"[{method}] a disabled plan node was needed:\n{sql}\n{plan}"
        problems = _unindexed_label_scans(plan)
        assert not problems, f"[{method}] label scan without an index:\n{sql}\n{plan}"
