"""The plan gate on a production-size copy of the graph (homelab#332, pre-Stage E).

``test_plans.py`` runs on small data with sequential scans off. At production
size the planner picks other plans, and the CI gate cannot see them. On
2026-11-10 two reads of the live and the delete path read a whole label at
production size (207,123 nodes, 847,841 edges):

* D2: ``get_filtered_graph_data([{type: [FactConflict]}])`` read every node with
  a Seq Scan, on every search.
* D3: ``find_edge_source_refs_by_document`` started from the GIN index of the
  dataset (651,981 edges), not from the nodes of the document. A small
  document took 138 s, and a large one passed the 5 min ``statement_timeout``.

This file runs those reads on a copy, with the default planner settings, and
limits the label rows that each statement reads by the size of its input:

* A statement with an id list (``$anchor_ids``, ``$ids``) may read
  ``FACTOR * (ids + the edges of those nodes) + SLACK`` label rows.
* A statement that reads the whole dataset by design (``DATASET_WIDE_PROBES``) may
  read each label once.
* Any other statement may read ``FACTOR * (rows it returns) + SLACK``.

⚠ "No Seq Scan" is not the check: the D3 plan was index-backed (a bitmap scan
of the GIN index) and still read the whole dataset.

Each statement is planned twice: with its parameters (a custom plan), and as
a prepared statement with ``plan_cache_mode = force_generic_plan``. psycopg
prepares a statement after its fifth run on one connection, and PostgreSQL can
then keep a generic plan.

Opt-in. Set ``AGE_COPY_GRAPH`` to the graph name of a copy; the connection
settings are those of ``conftest``. Without it, only the census runs: each
public read is probed or excluded by name. The inputs come from the graph
itself, so no id or name of the data is in this repository.

🚨 **Read-only.** This file calls no write method and never ``initialize()``.
Never point the ``adapter`` fixture or ``test_plans._workload`` at a copy:
that workload calls ``delete_graph``.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any, Callable, Dict, List

import psycopg
import pytest
from psycopg import sql

from cognee.infrastructure.databases.provenance import EdgeIdentity, parse_source_ref_key

from cognee_age_adapter.constants import EDGE_LABEL, NODE_LABEL

import conftest

pytestmark = [pytest.mark.integration, pytest.mark.production_copy, pytest.mark.asyncio]

COPY_GRAPH = os.getenv("AGE_COPY_GRAPH", "").strip()

# The allowed label rows for each unit of work, and a constant for the fixed
# lookups of a statement (the metadata vertex, an empty input).
FACTOR = 3
SLACK = 1000

# A statement that runs longer than this fails, as the production
# statement_timeout (5 min) fails a cognee call. Set for the adapter's own
# connections and for the EXPLAIN connections.
STATEMENT_TIMEOUT_MS = 120_000

# The statements that read the whole dataset by design: the statements of the
# per-document lookups that filter on the dataset GIN index. Each may read each
# label once.
# * The node pass reads every node of the dataset: the data id is only a
#   substring of a chunk-scope key, and no index serves CONTAINS.
# * The chunk sweep reads every edge of the dataset: create_chunk_associations
#   stamps a document's key on a chunk edge between OTHER documents' chunks.
DATASET_MARKER = "source_dataset_ids @> [$dataset_id]"
DATASET_WIDE_PROBES = frozenset({"document_smallest", "document_typical", "document_largest"})

LABEL_TABLES = (NODE_LABEL, EDGE_LABEL)
_ID = "ag_catalog.agtype_access_operator(VARIADIC ARRAY[{alias}.properties, '\"id\"'::ag_catalog.agtype])"

_inputs: Dict[str, Any] = {}


@pytest.fixture
def statement_timeout(monkeypatch):
    # libpq sends PGOPTIONS at connect, and a client option wins over the
    # role's own setting.
    monkeypatch.setenv("PGOPTIONS", f"-c statement_timeout={STATEMENT_TIMEOUT_MS}")


@pytest.fixture
async def copy_adapter(statement_timeout):
    if not COPY_GRAPH:
        pytest.skip("AGE_COPY_GRAPH names no production-size copy")
    instance = conftest.new_adapter(COPY_GRAPH)
    try:
        yield instance
    finally:
        await instance.close()


async def _discover(adapter) -> Dict[str, Any]:
    """The probe inputs, read from the copy once."""
    if _inputs:
        return _inputs
    try:
        chunks = await adapter.query(
            f"MATCH (n:{NODE_LABEL}) WHERE n.type = 'DocumentChunk' "
            "RETURN n.id AS id, n.source_ref_keys AS keys"
        )
    except (psycopg.OperationalError, OSError) as exc:
        conftest.unreachable(exc)
    # ORDER BY: the same inputs on every run.
    entities = await adapter.query(
        f"MATCH (n:{NODE_LABEL}) WHERE n.type = 'Entity' "
        "RETURN n.id AS id, n.source_ref_keys AS keys, n.source_run_ids AS runs "
        "ORDER BY n.id LIMIT 100"
    )
    assert len(chunks) >= 10 and len(entities) == 100, "the copy is not a production-size graph"
    entity_ids = [row["id"] for row in entities]
    # 20 edges of the node with the most outgoing edges: a read by edge identity
    # must not depend on the other edges of the source.
    with conftest.connect() as conn:
        hub = conn.execute(
            f"""
            SELECT {_ID.format(alias="n")}::text FROM "{COPY_GRAPH}"."{NODE_LABEL}" n
            WHERE n.id = (SELECT start_id FROM "{COPY_GRAPH}"."{EDGE_LABEL}"
                          GROUP BY start_id ORDER BY count(*) DESC, start_id LIMIT 1)
            """
        ).fetchone()[0]
    edges = await adapter.query(
        f"""
        MATCH (n:{NODE_LABEL})-[r:{EDGE_LABEL}]->(m:{NODE_LABEL}) WHERE n.id = $hub
        RETURN n.id AS s, m.id AS t, r.relationship_name AS rel
        ORDER BY t, rel LIMIT 20
        """,
        {"hub": hub},
    )
    sets = await adapter.query(
        f"""
        MATCH (s:{NODE_LABEL})-[r:{EDGE_LABEL}]->(n:{NODE_LABEL})
        WHERE n.type = 'NodeSet' AND r.relationship_name = 'belongs_to_set'
        RETURN n.name AS name, count(r) AS members
        """
    )

    sizes: Counter = Counter()
    for row in chunks:
        for key in row["keys"] or []:
            try:
                parsed = parse_source_ref_key(key)
            except ValueError:
                continue
            sizes[(str(parsed.dataset_id), str(parsed.data_id))] += 1
    # The documents of the dataset with the most chunks, by chunk count. The
    # largest document of the copy can be the only document of its own
    # dataset, where an index on the dataset costs no more than the document.
    by_dataset: Counter = Counter()
    for (dataset_id, _), count in sizes.items():
        by_dataset[dataset_id] += count
    biggest = by_dataset.most_common(1)[0][0]
    ranked = [
        document
        for document, _ in sorted(sizes.items(), key=lambda item: (item[1], item[0]))
        if document[0] == biggest
    ]
    _inputs.update(
        chunks=sorted(row["id"] for row in chunks)[:10],
        entities=entity_ids,
        edges=[EdgeIdentity(row["s"], row["t"], row["rel"]) for row in edges],
        source_ref_key=next(row["keys"][0] for row in entities if row["keys"]),
        run_id=next((row["runs"][0] for row in entities if row["runs"]), None),
        # A set of approximately 1,000 members: a scope read, not the whole graph.
        node_set=min(sets, key=lambda row: abs(row["members"] - 1000))["name"],
        documents={
            "smallest": ranked[0],
            "typical": ranked[len(ranked) // 2],
            "largest": ranked[-1],
        },
    )
    assert _inputs["run_id"], "no Entity of the sample has a pipeline run id"
    return _inputs


def _document(which: str) -> Callable:
    async def call(adapter, inputs):
        dataset_id, data_id = inputs["documents"][which]
        await adapter.find_node_source_refs_by_document(dataset_id, data_id)
        await adapter.find_edge_source_refs_by_document(dataset_id, data_id)

    return call


async def _node_reads(adapter, inputs) -> None:
    one, ids = inputs["entities"][0], inputs["entities"][:20]
    await adapter.is_empty()
    await adapter.has_node(one)
    await adapter.get_node(one)
    await adapter.get_nodes(ids)
    await adapter.get_edges(one)
    await adapter.get_neighbors(one)
    await adapter.get_connections(one)
    await adapter.get_node_truth_state(ids)
    await adapter.get_node_delete_data(ids)


async def _edge_reads(adapter, inputs) -> None:
    edges = inputs["edges"]
    triples = [(edge.source_id, edge.target_id, edge.relationship_name) for edge in edges]
    await adapter.has_edge(*triples[0])
    await adapter.has_edges([(*triple, {}) for triple in triples])
    await adapter.get_edge_delete_data(edges)


async def _source_ref_reads(adapter, inputs) -> None:
    await adapter.find_nodes_by_source_ref(inputs["source_ref_key"])
    await adapter.find_edges_by_source_ref(inputs["source_ref_key"])
    await adapter.find_node_source_refs_by_pipeline_run(inputs["run_id"])
    await adapter.find_edge_source_refs_by_pipeline_run(inputs["run_id"])


async def _node_set_scope(adapter, inputs) -> None:
    from cognee.modules.engine.models.node_set import NodeSet

    await adapter.get_nodeset_subgraph(NodeSet, [inputs["node_set"]])


_DOCUMENT_METHODS = frozenset(
    {"find_node_source_refs_by_document", "find_edge_source_refs_by_document"}
)

# {probe: (the public methods it calls, the call)}
PROBES: Dict[str, tuple] = {
    # Every search reads the FactConflict nodes (cognee 1.6.7, conflicts).
    "filter_by_one_type": (
        {"get_filtered_graph_data"},
        lambda adapter, inputs: adapter.get_filtered_graph_data(
            [{"type": ["FactConflict"]}], include_edges=False
        ),
    ),
    # The conflict resolution reads one node by id this way.
    "filter_by_one_id": (
        {"get_filtered_graph_data"},
        lambda adapter, inputs: adapter.get_filtered_graph_data(
            [{"id": [inputs["entities"][0]]}], include_edges=False
        ),
    ),
    # The memify tasks read a type with its edges.
    "filter_with_edges": (
        {"get_filtered_graph_data"},
        lambda adapter, inputs: adapter.get_filtered_graph_data([{"type": ["TextDocument"]}]),
    ),
    # Every RAG and CHUNKS search: the conflict citations of the hits.
    "conflict_neighbourhood": (
        {"get_neighborhood"},
        lambda adapter, inputs: adapter.get_neighborhood(
            inputs["chunks"], depth=1, edge_types=["conflict_cites"]
        ),
    ),
    # GRAPH_COMPLETION with a neighbourhood: every edge among the reached nodes.
    "neighbourhood_depth_1": (
        {"get_neighborhood"},
        lambda adapter, inputs: adapter.get_neighborhood(inputs["chunks"], depth=1),
    ),
    # GRAPH_COMPLETION: the projection of the vector hits.
    "id_filtered_graph": (
        {"get_id_filtered_graph_data"},
        lambda adapter, inputs: adapter.get_id_filtered_graph_data(inputs["entities"]),
    ),
    # GRAPH_COMPLETION with a nodeName only.
    "node_set_scope": ({"get_nodeset_subgraph"}, _node_set_scope),
    "node_reads": (
        {
            "is_empty", "has_node", "get_node", "get_nodes", "get_edges", "get_neighbors",
            "get_connections", "get_node_truth_state", "get_node_delete_data",
        },
        _node_reads,
    ),
    "edge_reads": ({"has_edge", "has_edges", "get_edge_delete_data"}, _edge_reads),
    "source_ref_reads": (
        {
            "find_nodes_by_source_ref", "find_edges_by_source_ref",
            "find_node_source_refs_by_pipeline_run", "find_edge_source_refs_by_pipeline_run",
        },
        _source_ref_reads,
    ),
    "graph_metadata": ({"get_graph_metadata"}, lambda adapter, inputs: adapter.get_graph_metadata()),
    # The per-document delete and the prune.
    "document_smallest": (_DOCUMENT_METHODS, _document("smallest")),
    "document_typical": (_DOCUMENT_METHODS, _document("typical")),
    "document_largest": (_DOCUMENT_METHODS, _document("largest")),
}

# The public methods that no probe calls, and why.
NOT_PROBED = {
    "get_graph_data": "reads the whole graph by design",
    "get_graph_metrics": "reads the whole graph by design",
    "get_triplets_batch": "pages through the whole graph by design",
    "find_node_source_refs_by_dataset": "reads the whole dataset by design (delete_by_dataset)",
    "find_edge_source_refs_by_dataset": "reads the whole dataset by design (delete_by_dataset)",
}

# The write methods: never on a copy (the CI gate checks their plans).
WRITES = frozenset(
    {
        "add_node", "add_nodes", "add_edge", "add_edges", "delete_node", "delete_nodes",
        "delete_edge_triples", "delete_graph", "remove_belongs_to_set_tags",
        "attach_node_source_refs", "attach_edge_source_refs", "remove_node_source_refs",
        "remove_edge_source_refs", "set_graph_metadata", "set_node_truth_state",
    }
)


def test_every_public_read_is_probed_or_excluded_by_name():
    """🚨 A new public read fails here until a probe calls it or NOT_PROBED names it."""
    from test_plans import _public_data_methods

    probed = frozenset().union(*(methods for methods, _ in PROBES.values()))
    reads = _public_data_methods() - WRITES
    assert not probed & frozenset(NOT_PROBED), "a method is both probed and excluded"
    assert probed | frozenset(NOT_PROBED) == reads, (
        f"not probed and not excluded: {sorted(reads - probed - frozenset(NOT_PROBED))}; "
        f"not a public read: {sorted((probed | frozenset(NOT_PROBED)) - reads)}"
    )


async def _record(adapter, call) -> List[tuple]:
    sent = conftest.record_sql(adapter)
    try:
        await call
    finally:
        del adapter._execute
    return [(text, args) for text, args in sent if text.startswith("SELECT * FROM ag_catalog.cypher")]


def _cypher_text(statement: str) -> str:
    return statement.split("$cypher$")[1]


def _explain(conn: psycopg.Connection, statement: str, args, generic: bool) -> dict:
    """``EXPLAIN (ANALYZE, BUFFERS)`` of one recorded statement, as a custom or a generic plan."""
    options = "ANALYZE, BUFFERS, FORMAT JSON"
    if not generic:
        return conn.execute(f"EXPLAIN ({options}) {statement}", args).fetchone()[0][0]
    # The recorded SQL binds the parameter map as the last ``%s``, and doubles
    # every ``%`` of the Cypher text for psycopg. A PREPARE is sent without
    # parameters, so the text is sent as it is: undo both.
    head, tail = statement.rsplit(", %s) AS (", 1)
    prepared = (head + ", $1) AS (" + tail).replace("%%", "%")
    conn.execute("SET plan_cache_mode = force_generic_plan")
    conn.execute(f"PREPARE probe(ag_catalog.agtype) AS {prepared}")
    try:
        # A literal, not a parameter: EXECUTE takes its arguments as expressions.
        execute = sql.SQL("EXPLAIN ({}) EXECUTE probe({})").format(
            sql.SQL(options), sql.Literal(args[0])
        )
        return conn.execute(execute).fetchone()[0][0]
    finally:
        conn.execute("DEALLOCATE probe")
        conn.execute("RESET plan_cache_mode")


def _label_rows(node: dict) -> int:
    """The rows that the scans of a label table read, filtered rows included."""
    total = 0
    if node.get("Relation Name") in LABEL_TABLES:
        per_loop = (
            node.get("Actual Rows", 0)
            + node.get("Rows Removed by Filter", 0)
            + node.get("Rows Removed by Index Recheck", 0)
        )
        total += round(per_loop * node.get("Actual Loops", 0))
    for child in node.get("Plans", []):
        total += _label_rows(child)
    return total


def _incident_edges(conn: psycopg.Connection, ids: List[str]) -> int:
    """The nodes with these ids plus their edges in both directions, counted on graphids."""
    graph = COPY_GRAPH
    row = conn.execute(
        f"""
        WITH wanted AS (SELECT to_json(x)::text::ag_catalog.agtype AS v FROM unnest(%s::text[]) AS x),
        anchors AS MATERIALIZED (
            SELECT n.id AS gid FROM "{graph}"."{NODE_LABEL}" n
            JOIN wanted ON {_ID.format(alias="n")} = wanted.v)
        SELECT (SELECT count(*) FROM anchors)
             + (SELECT count(*) FROM "{graph}"."{EDGE_LABEL}" e JOIN anchors a ON e.start_id = a.gid)
             + (SELECT count(*) FROM "{graph}"."{EDGE_LABEL}" e JOIN anchors a ON e.end_id = a.gid)
        """,
        (list(ids),),
    ).fetchone()
    return int(row[0])


def _whole_labels(conn: psycopg.Connection) -> int:
    row = conn.execute(
        "SELECT sum(c.reltuples)::bigint FROM pg_class c JOIN pg_namespace s ON s.oid = c.relnamespace "
        "WHERE s.nspname = %s AND c.relname = ANY(%s)",
        (COPY_GRAPH, list(LABEL_TABLES)),
    ).fetchone()
    return int(row[0])


def _budget(conn, probe: str, cypher: str, params: dict, returned: int) -> tuple[int, str]:
    ids = params.get("anchor_ids", params.get("ids"))
    if ids is not None:
        work = _incident_edges(conn, ids)
        return FACTOR * work + SLACK, f"{len(ids)} ids, {work} nodes and edges"
    if DATASET_MARKER in cypher and probe in DATASET_WIDE_PROBES:
        return _whole_labels(conn), "dataset-wide by design"
    return FACTOR * returned + SLACK, f"{returned} rows returned"


# Known defects that this gate found, open in homelab#332. Strict: a fix makes
# the probe pass, and then this entry must go.
KNOWN_DEFECTS = {
    "edge_reads": (
        "A read by edge identity (_edges_by_identity) reads every outgoing edge of "
        "each source through the start_id index, and filters the name and the "
        "target after the join. On the copy (2026-11-10), one edge of the source "
        "with the most edges read 1,273 label rows, and 20 of its edges read 25,551. "
        "The edge identity index does not serve it."
    ),
}


@pytest.mark.parametrize(
    "probe",
    [
        pytest.param(name, marks=pytest.mark.xfail(strict=True, reason=KNOWN_DEFECTS[name]))
        if name in KNOWN_DEFECTS
        else name
        for name in sorted(PROBES)
    ],
)
async def test_each_read_follows_its_input_at_production_size(copy_adapter, probe):
    inputs = await _discover(copy_adapter)
    _, call = PROBES[probe]
    statements = await _record(copy_adapter, call(copy_adapter, inputs))
    assert statements, f"{probe} sent no Cypher statement"

    problems = []
    with conftest.connect() as conn:
        for statement, args in statements:
            cypher = _cypher_text(statement)
            params = json.loads(args[0])
            for generic in (False, True):
                plan = _explain(conn, statement, args, generic)
                read = _label_rows(plan["Plan"])
                returned = round(plan["Plan"].get("Actual Rows", 0))
                budget, basis = _budget(conn, probe, cypher, params, returned)
                if read > budget:
                    kind = "generic" if generic else "custom"
                    problems.append(
                        f"{kind} plan read {read} label rows, budget {budget} ({basis}), "
                        f"{plan['Execution Time']:.0f} ms:\n{' '.join(cypher.split())}"
                    )
    assert not problems, f"[{probe}]\n" + "\n\n".join(problems)
