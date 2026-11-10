"""Apache AGE graph adapter for cognee.

A port of ``sirantd/cognee-falkordb-adapter`` 0.5.4 (homelab#332, Stage A). The
public surface, the provenance semantics and the value rules are the same. The
store layout is new, and the reasons are measurements of AGE 1.8.0 on
2026-11-08 (``apache/age:release_PG18_1.8.0``, arm64):

* The graph has one vertex label and one edge label. The cognee type is the
  ``type`` property, and the relationship name is the ``relationship_name``
  property. ``constants.py`` has the measurement.
* Every lookup by a value is ``UNWIND $values AS v ... WHERE n.p = v``. ⚠ The
  feasibility notes of the issue say that a btree index serves ``IN``. It does
  not: ``WHERE n.id IN $ids`` compiles to ``agtype_in_operator(...)``, a filter
  that no index serves. ``IN`` stays only as a filter after an index-backed step.
* A vertex write uses no ``MERGE``. AGE matches a MERGE pattern with
  ``properties @> {id: ...}``, and only a GIN index over ALL properties serves
  that. ``add_nodes`` creates the absent ids, then updates every node through
  the unique ``id`` index, in one transaction.
* Every edge read is directed. An undirected ``(a)-[r]-(b)`` plans as a join
  with an ``OR`` condition that reads every vertex. Each undirected read here
  is two directed reads joined by ``UNION``.
* Unique indexes on the node id and on the edge identity
  ``(start_id, end_id, relationship_name)`` stop duplicates.
* Every node and edge write takes a transaction-level advisory lock for each
  node id or edge identity first, in sorted order (homelab#332, D4). AGE fails
  a SET on a row that a concurrent transaction updated, so two writers of one
  node must not overlap. A write that still loses a race runs again.

Two rules come from the plans at production size (207,123 nodes, 847,841
edges, measured 2026-11-10). The CI plan gate cannot see them; the gate of
``tests/test_plans_production.py`` runs on a copy:

* PostgreSQL estimates 1,000 rows for every ``UNWIND`` of a parameter. For one
  value, that estimate picks a Seq Scan and a hash join over the whole label,
  so a filter with one value is ``n.p = $value`` (D2).
* A read that starts from given nodes must not also carry a condition that an
  index serves for many rows. The planner underestimates a GIN ``@>`` on the
  dataset (818 rows estimated, 651,981 actual) and starts from it (D3).

A third rule comes from the same copy (D5). The planner estimates 9 edges
for each source, so every Cypher form of a match by edge identity reads all
outgoing edges of the source through AGE's ``start_id`` index (1,271 for the
largest source). So a SQL statement finds the graphid of each edge, with both
ends known before the edge scan, and the Cypher statement matches ``id(r)``.
The index of that edge scan is a cost decision; ``_edge_id_sql`` has the
margin.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
from contextlib import asynccontextmanager, nullcontext
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    LiteralString,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
    cast,
)
from uuid import UUID

import psycopg
from psycopg import errors
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg_pool import AsyncConnectionPool, PoolTimeout

from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
from cognee.infrastructure.databases.provenance import (
    EdgeDeleteData,
    EdgeIdentity,
    NodeDeleteData,
    get_dataset_id_from_source_ref_key,
    get_pipeline_run_id_from_source_run_ref,
    get_source_ref_key_from_source_run_ref,
    parse_source_ref_key,
)
from cognee.infrastructure.databases.provenance.source_ref_state import (
    provenance_after_attach,
    provenance_after_remove,
    provenance_attach_inputs,
)
from cognee.infrastructure.engine import DataPoint
from cognee.shared.logging_utils import get_logger

from . import agtype
from .coercion import coerce_properties, scrub_nul
from .constants import (
    ANCHOR_BATCH,
    CHUNK_TYPE,
    DEFAULT_GRAPH_NAME,
    DEFAULT_HOST,
    DEFAULT_PORT,
    EDGE_GIN_INDEXES,
    EDGE_LABEL,
    LOCK_BATCH,
    METADATA_LABEL,
    METADATA_NODE_ID,
    NODE_BTREE_INDEXES,
    NODE_GIN_INDEXES,
    NODE_LABEL,
    POOL_MAX_SIZE,
    POOL_MIN_SIZE,
    PROVENANCE_COLUMNS,
    RELATIONSHIP_PROPERTY,
    WRITE_ATTEMPTS,
)
from .cypher import AGTYPE, compose, return_columns, validate_graph_name

logger = get_logger("AGEAdapter")

Node = Any
NodeData = Dict[str, Any]
EdgeData = Tuple[str, str, str, Dict[str, Any]]

_PROVENANCE_KEY_SET = frozenset(PROVENANCE_COLUMNS)

# The vertex properties with a btree index. A filtered read on one of them is an
# index scan; on any other property it reads the whole label.
_INDEXED_NODE_PROPERTIES = frozenset(name for name, _ in NODE_BTREE_INDEXES)

# The errors of a write that lost a race to a concurrent writer. The write is
# idempotent, so it runs again: the second time it finds what the winner made.
_RETRIED_ERRORS = (errors.UniqueViolation, errors.DeadlockDetected, errors.SerializationFailure)

# 🚨 AGE 1.8.0 (src/backend/executor/cypher_set.c): a SET on a vertex or an edge
# that a concurrent transaction updated waits for that commit, and then FAILS
# with XX000 "Entity failed to be updated: 3" (TM_Updated). A native UPDATE
# would re-check the row instead. Measured with two writers on one graph.
_AGE_UPDATE_CONFLICT = "Entity failed to be updated"


def _lost_a_race(exc: Exception) -> bool:
    """True if ``exc`` is a write that lost a race and can run again."""
    if isinstance(exc, _RETRIED_ERRORS):
        return True
    return isinstance(exc, errors.InternalError_) and str(exc).startswith(_AGE_UPDATE_CONFLICT)


async def _wait_before_retry(attempt: int, exc: Exception) -> None:
    logger.info(
        "Write lost a race (%s), attempt %d of %d", type(exc).__name__, attempt, WRITE_ATTEMPTS
    )
    await asyncio.sleep(0.02 * attempt + random.uniform(0, 0.02))


# 🚨 homelab#332 D4. Measured on production 2026-11-10: with 5 writers, 9 of
# 80 batches failed after WRITE_ATTEMPTS, because each retry met the next
# writer. So a node or edge write first waits for every other writer of the
# same nodes or edges: one transaction-level advisory lock for each node id or
# edge identity, in sorted order (two writers cannot wait on each other), in
# the first statement. The Cypher statements after it read the rows only
# after the other writer committed. The lock ends with the transaction.
#
# ⚠ The rows come out of unnest() in array order, and the locks are taken in
# that order: a Function Scan with no other node returns its rows in order.
_LOCK_SQL = "SELECT pg_catalog.pg_advisory_xact_lock(key) FROM unnest(%s::bigint[]) AS key"


def _lock_key(graph: str, kind: str, *parts: str) -> int:
    """A stable 64-bit advisory lock key for one node or one edge of one graph.

    Not ``hash()``: Python salts it for each process, and the key must be equal
    in every writer. A collision of two keys only makes two writers wait.
    """
    text = "\x00".join((graph, kind, *parts)).encode()
    return int.from_bytes(hashlib.blake2b(text, digest_size=8).digest(), "big", signed=True)


def _slices(items: Sequence[Any], size: int) -> List[Sequence[Any]]:
    """``items`` in slices of at most ``size``, in their order."""
    return [items[start : start + size] for start in range(0, len(items), size)]


def _in_lock_batches(items: Sequence[Any]) -> List[Sequence[Any]]:
    return _slices(items, LOCK_BATCH)


# How long the first connection may take before initialize() reports the cause.
_POOL_OPEN_TIMEOUT = 15.0

N = NODE_LABEL
E = EDGE_LABEL
REL = RELATIONSHIP_PROPERTY


def _property(prop: str, alias: str = "") -> str:
    """The SQL expression that AGE compiles ``x.<prop>`` into.

    Every property index is on this expression, so a SQL condition must use it
    exactly, or the planner cannot match the index.
    """
    column = f"{alias}.properties" if alias else "properties"
    literal = json.dumps(prop).replace("'", "''")
    return f"ag_catalog.agtype_access_operator(VARIADIC ARRAY[{column}, '{literal}'::{AGTYPE}])"


def _edge_id_sql(graph: str) -> str:
    """SQL that finds the graphid of each edge identity, or NULL if it is absent.

    🚨 homelab#332 D5. Not Cypher: AGE compiles every Cypher form of a match by
    identity to a join, and the planner then reads the edge through the
    ``start_id`` index of the source with the name and the target as filters.
    Measured on a production-size copy (2026-11-10): 20 edges of the source
    with 1,271 outgoing edges read 25,551 label rows.

    Here each end is a scalar subquery on the unique ``id`` index, so both
    ends are known before the edge scan: the join order is fixed. ⚠ The index
    of the edge scan is still a cost decision, because AGE's ``start_id`` and
    ``end_id`` indexes also serve the condition. On the copy the planner
    estimates 9 edges for each source and 72 for each target. So the identity
    index (cost 25.4, one row) wins over a BitmapAnd of the two AGE indexes
    (30.7) and over either one alone (approximately 57). The custom and the
    generic plan read three label rows for each edge. When each end has
    approximately one edge, the costs tie, and the planner can pick any of the
    three; ``test_plans.py`` and ``tests/test_plans_production.py`` assert the
    index.

    The parameters are three text arrays: the source ids, the target ids and
    the relationship names. Each row is ``(position, graphid)``, with the
    position from 1.
    """

    def node_gid(alias: str, value: str) -> str:
        return (
            f'(SELECT {alias}.id FROM "{graph}"."{N}" {alias} '
            f"WHERE {_property('id', alias)} = to_json({value})::text::{AGTYPE})"
        )

    return f"""
        SELECT w.position, (
            SELECT r.id::text FROM "{graph}"."{E}" r
            WHERE r.start_id = {node_gid("a", "w.s")}
              AND r.end_id = {node_gid("b", "w.t")}
              AND {_property(REL, "r")} = to_json(w.rel)::text::{AGTYPE}
        )
        FROM unnest(%s::text[], %s::text[], %s::text[]) WITH ORDINALITY AS w(s, t, rel, position)
    """


# Binds ``e`` to one row of ``$edges``, which carries the graphid ``id`` from
# ``_edge_id_sql``, and ``a``, ``r`` and ``b`` to the edge and its ends. Every
# step is a lookup of one row: ``edge_id_btree``, then the vertex primary key.
_EDGES_BY_ID = f"""UNWIND $edges AS e
            MATCH (a:{N})-[r:{E}]->(b:{N}) WHERE id(r) = e.id"""


class _Session:
    """The statements of one open transaction.

    A call runs one Cypher statement: ``(query, params) -> rows``.
    """

    def __init__(self, adapter: "AGEAdapter", cursor: psycopg.AsyncCursor):
        self._adapter = adapter
        self._cursor = cursor

    async def __call__(self, query: str, params: Optional[dict] = None) -> List[Dict[str, Any]]:
        return await self._adapter._cypher(self._cursor, query, params)

    async def edge_ids(self, rows: Sequence[dict]) -> List[Optional[int]]:
        """The graphid of the edge of each row (``s``, ``t``, ``rel``), or None."""
        if not rows:
            return []
        # NUL is scrubbed as _cypher scrubs it, so a key finds what a write stored.
        columns = [[scrub_nul(str(row[key])) for row in rows] for key in ("s", "t", "rel")]
        await self._adapter._execute(self._cursor, self._adapter._edge_id_sql, columns)
        found = {position: gid for position, gid in await self._cursor.fetchall()}
        return [
            int(gid) if (gid := found.get(position)) is not None else None
            for position in range(1, len(rows) + 1)
        ]

    async def on_edges(self, rows: Sequence[dict], tail: str) -> List[Dict[str, Any]]:
        """Run ``tail`` once for each row whose edge exists, matched by identity.

        Each row is a map with ``s``, ``t`` and ``rel`` (``_edge_identity_row``),
        and with the other values that ``tail`` reads from ``e``. ``tail`` sees
        ``e``, ``a``, ``r`` and ``b`` (``_EDGES_BY_ID``). A row whose edge is
        absent is dropped, as the match by identity dropped it.
        """
        ids = await self.edge_ids(rows)
        found = [{**row, "id": gid} for row, gid in zip(rows, ids) if gid is not None]
        if not found:
            return []
        return await self(f"{_EDGES_BY_ID}\n{tail}", {"edges": found})


def _strip_provenance(properties: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a copy of ``properties`` without the four provenance list properties.

    Provenance is storage-internal: it must not reach retrieval, DataPoint
    reconstruction or the ``*_properties`` field of the delete-data payload,
    which carries the four fields in their own typed slots.
    """
    return {
        key: value
        for key, value in (properties or {}).items()
        if key not in _PROVENANCE_KEY_SET
    }


def _prov_list(value: Any) -> List[str]:
    """Normalize a native provenance list property (or a missing one) to ``list[str]``."""
    return list(value) if value else []


def _document_refs(keys: Iterable[str], dataset_id: str, data_id: str) -> List[str]:
    """The keys owned by one document of one dataset — v1 doc-scope and v2 chunk-scope.

    The exact half of the per-document lookups. The Cypher pre-filter
    (``CONTAINS $data_id``) is a superset: it matches the id in ANY segment, and
    the keys of the same document under another dataset. A key that does not
    decompose is skipped, as cognee's ``delete_by_document`` filter skips it.
    """
    owned = []
    for key in keys:
        try:
            parsed = parse_source_ref_key(key)
        except ValueError:
            continue
        if str(parsed.dataset_id) == dataset_id and str(parsed.data_id) == data_id:
            owned.append(key)
    return owned


def _quote(identifier: str) -> str:
    """Backtick-quote a property name for interpolation into Cypher.

    A property name cannot be a parameter, so ``get_filtered_graph_data`` puts
    it into the query text. AGE reads two backticks inside a quoted identifier
    as one backtick (measured). So a name with a backtick still names its own
    key, and it cannot end the identifier. NUL is removed, because the query
    text cannot hold it.
    """
    cleaned = scrub_nul(identifier)
    if cleaned != identifier:
        logger.warning("Dropped NUL from graph identifier %r", identifier)
    return "`" + cleaned.replace("`", "``") + "`"


def _dedupe(values: Iterable[Any]) -> List[Any]:
    """Order-preserving dedupe."""
    return list(dict.fromkeys(values))


def _provenance_fold_clause(alias: str) -> str:
    """Cypher ``SET`` fragment that stamps provenance inside the artifact write.

    Appended to the ``add_nodes`` / ``add_edges`` update, so an artifact is
    written and stamped in one statement, with no read-then-write window. The
    run ref and id are appended only when the key is *not* already present
    (Model A).

    Every guard reads ``prov_keys_before``, the pre-mutation
    ``source_ref_keys`` that the caller binds in a preceding ``WITH``, so the
    clause does not depend on the evaluation order of a SET list.
    """
    return f"""
            SET {alias}.source_run_refs = CASE
                    WHEN $prov_sr_key IN prov_keys_before
                    THEN coalesce({alias}.source_run_refs, [])
                    ELSE coalesce({alias}.source_run_refs, []) + $prov_add_run_refs
                END,
                {alias}.source_run_ids = CASE
                    WHEN $prov_sr_key IN prov_keys_before
                    THEN coalesce({alias}.source_run_ids, [])
                    ELSE coalesce({alias}.source_run_ids, [])
                         + [x IN $prov_add_run_ids
                            WHERE NOT x IN coalesce({alias}.source_run_ids, [])]
                END,
                {alias}.source_ref_keys = CASE
                    WHEN $prov_sr_key IN prov_keys_before
                    THEN prov_keys_before
                    ELSE prov_keys_before + $prov_add_keys
                END,
                {alias}.source_dataset_ids = CASE
                    WHEN $prov_ds_id IN coalesce({alias}.source_dataset_ids, [])
                    THEN coalesce({alias}.source_dataset_ids, [])
                    ELSE coalesce({alias}.source_dataset_ids, []) + $prov_add_dataset_ids
                END
            """


def _provenance_fold_params(source_ref_key: str, pipeline_run_id: Optional[str]) -> Dict[str, Any]:
    """Scalar query params consumed by :func:`_provenance_fold_clause`."""
    inputs = provenance_attach_inputs(source_ref_key, pipeline_run_id)
    return {
        "prov_sr_key": inputs.source_ref_key,
        "prov_add_keys": list(inputs.add_keys),
        "prov_ds_id": inputs.add_dataset_ids[0],
        "prov_add_dataset_ids": list(inputs.add_dataset_ids),
        "prov_add_run_refs": list(inputs.add_run_refs),
        "prov_add_run_ids": list(inputs.add_run_ids),
    }


def _connected_components(node_ids: Sequence[str], edges: Sequence[Tuple[str, str]]) -> List[int]:
    """Component sizes of an undirected graph, by union-find.

    Kept in Python: an unbounded variable-length traversal for each node would
    not finish on a graph of 200k nodes. Two id-only reads and this are exact.
    """
    parent: Dict[str, str] = {node_id: node_id for node_id in node_ids}

    def find(item: str) -> str:
        root = item
        while parent[root] != root:
            root = parent[root]
        while parent[item] != root:  # path compression
            parent[item], item = root, parent[item]
        return root

    for source, target in edges:
        if source not in parent or target not in parent:
            continue
        source_root, target_root = find(source), find(target)
        if source_root != target_root:
            parent[source_root] = target_root

    sizes: Dict[str, int] = {}
    for node_id in parent:
        root = find(node_id)
        sizes[root] = sizes.get(root, 0) + 1
    return sorted(sizes.values(), reverse=True)


def _edge_tuple(row: Dict[str, Any]) -> EdgeData:
    """A ``(source, target, relationship_name, properties)`` tuple from an edge row.

    ⚠ Falls back to the matched endpoints when the edge has no
    ``source_node_id`` / ``target_node_id``: an edge that a bulk loader wrote
    can lack them.
    """
    properties = _strip_provenance(row["properties"])
    return (
        properties.get("source_node_id", row["source"]),
        properties.get("target_node_id", row["target"]),
        row["type"],
        properties,
    )


def _node_tuples(rows: List[Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    return [
        (properties["id"], properties)
        for properties in (_strip_provenance(row["properties"]) for row in rows)
    ]


def _anchored_edges_query(filters: str = "", returns: Optional[str] = None) -> str:
    """Both directions of the edges of the nodes with ``id IN $anchor_ids``.

    Two directed, index-backed reads joined by ``UNION``: from each anchor
    through the ``start_id`` index, and through the ``end_id`` index. ``UNION``
    removes the second copy of an edge whose two endpoints are both anchors.
    ``filters`` is an extra ``WHERE`` condition on ``r``, and ``returns`` the
    ``RETURN`` items, with ``s``, ``r`` and ``t`` bound to the edge as stored.
    """
    returns = returns or (
        f"s.id AS source, t.id AS target, r.{REL} AS type, properties(r) AS properties"
    )
    condition = f" AND {filters}" if filters else ""
    return f"""
        UNWIND $anchor_ids AS anchor
        MATCH (s:{N})-[r:{E}]->(t:{N}) WHERE s.id = anchor{condition}
        RETURN {returns}
        UNION
        UNWIND $anchor_ids AS anchor
        MATCH (s:{N})-[r:{E}]->(t:{N}) WHERE t.id = anchor{condition}
        RETURN {returns}
    """


class AGEAdapter(GraphDBInterface):
    """cognee graph adapter backed by an Apache AGE graph."""

    def __init__(
        self,
        graph_database_url: Optional[str] = None,
        graph_database_username: Optional[str] = None,
        graph_database_password: Optional[str] = None,
        graph_database_port: Union[int, str, None] = None,
        graph_database_key: Optional[str] = None,
        database_name: Optional[str] = None,
        graph_database_name: Optional[str] = None,
        host: Optional[str] = None,
        port: Union[int, str, None] = None,
        postgres_database: Optional[str] = None,
    ):
        """Prepare a connection pool to PostgreSQL and select the AGE graph.

        🚨 **The first six keyword names are cognee's, not ours.** A registered
        adapter is built by ``_create_graph_engine`` as::

            adapter(graph_database_url=…, graph_database_username=…,
                    graph_database_password=…, graph_database_port=…,
                    graph_database_key=…, database_name=…)

        so every one of those six must bind, or the first cognify, recall or
        delete fails with a ``TypeError``. ``test_factory.py`` reads that call
        from cognee's own source.

        * ``graph_database_url`` is a host, as cognee's other client-server
          stores take it, or a full ``postgresql://`` URL.
        * ``database_name`` (cognee's ``GRAPH_DATABASE_NAME``) is the AGE graph
          name. The PostgreSQL database is ``postgres_database``, then the
          ``dbname`` of a URL, and else the graph name. So production sets one
          name, ``cognee_graph``, for the database and for the graph.
        * ``graph_database_key`` has no PostgreSQL meaning. cognee passes it
          always, so it is accepted, and a value that is set is warned about.

        📌 The port is **not** changed from cognee's unset sentinel (``123``). A
        missing ``GRAPH_DATABASE_PORT`` is then a refused connection on port 123,
        and not a connection to another server.

        No connection is made here. The pool opens on the first query, inside
        cognee's event loop, and ``initialize()`` reports a connection failure
        with its real cause.
        """
        # cognee passes "" for anything unset, so falsiness, not None, separates
        # "not configured" from "configured".
        self._graph_name = validate_graph_name(
            database_name or graph_database_name or DEFAULT_GRAPH_NAME
        )
        self._edge_id_sql = _edge_id_sql(self._graph_name)

        if graph_database_key:
            logger.warning(
                "graph_database_key is set but PostgreSQL has no such credential; ignoring it. "
                "Use graph_database_username / graph_database_password instead."
            )

        resolved_host = host or graph_database_url or DEFAULT_HOST
        url = str(resolved_host) if "://" in str(resolved_host) else ""
        explicit_port = port or graph_database_port
        settings: Dict[str, Any] = {}
        if not url:
            settings["host"] = resolved_host
            settings["port"] = int(explicit_port or DEFAULT_PORT)
        elif explicit_port:
            settings["port"] = int(explicit_port)
        if graph_database_username:
            settings["user"] = graph_database_username
        if graph_database_password:
            settings["password"] = graph_database_password
        if postgres_database:
            settings["dbname"] = postgres_database
        elif not conninfo_to_dict(url).get("dbname"):
            settings["dbname"] = self._graph_name
        # 🚨 No LOAD 'age' and no search_path here: the graph role's own
        # settings preload AGE and set the search path. A LOAD from a
        # non-superuser fails with `access to library "age" is not allowed`.
        # The search path is load-bearing. Measured: without ag_catalog in it,
        # AGE's own label DDL fails (`operator class "graphid_ops" does not
        # exist`). A client setting here would hide a role that lacks it.
        settings["application_name"] = "cognee-age-adapter"

        self._conninfo = make_conninfo(url, **settings)
        self._pool = self._new_pool()
        self._pool_lock = asyncio.Lock()
        self._pool_open = False
        # Serializes the read-modify-write of the explicit attach/remove
        # source-ref calls, and the folded writes that race them.
        self._source_ref_change_lock = asyncio.Lock()
        # Set once initialize() has made the graph, its labels and its indexes.
        self._initialized = False

    # ------------------------------------------------------------------
    # Connection and statements
    # ------------------------------------------------------------------

    def _new_pool(self) -> AsyncConnectionPool:
        """Make a pool that is not open yet.

        🚨 psycopg_pool cannot open a pool again after a close. Each close puts
        a new pool in its place, or every later call fails with ``PoolClosed``.
        """
        return AsyncConnectionPool(
            self._conninfo,
            min_size=POOL_MIN_SIZE,
            max_size=POOL_MAX_SIZE,
            open=False,
            kwargs={"autocommit": True},
            # The shared PostgreSQL app restarts for its own maintenance. A
            # connection from before a restart is replaced, not handed out dead.
            check=AsyncConnectionPool.check_connection,
            name=f"cognee-age-{self._graph_name}",
        )

    async def _open_pool(self) -> None:
        if self._pool_open:
            return
        async with self._pool_lock:
            if self._pool_open:
                return
            try:
                await self._pool.open(wait=True, timeout=_POOL_OPEN_TIMEOUT)
            except PoolTimeout:
                # The server can be down for a restart. The new pool lets the
                # next call try again. 🚨 Replace it before the probe below,
                # because the probe raises.
                await self._pool.close()
                self._pool = self._new_pool()
                # The pool only says "initialization incomplete". One direct
                # connection raises the real cause: a wrong host, password or
                # database, or a role without the AGE preload.
                async with await psycopg.AsyncConnection.connect(self._conninfo):
                    pass
                raise
            self._pool_open = True

    async def _execute(self, cursor: psycopg.AsyncCursor, sql: str, args: Any) -> None:
        """The one place that sends SQL. The plan tests record what passes here.

        The SQL is composed, not literal: ``compose()`` validates the graph name
        and keeps the Cypher text inside its dollar quotes, and every value is
        a bound parameter. The cast says so to the type checker.
        """
        await cursor.execute(cast(LiteralString, sql), args)

    async def _cypher(
        self, cursor: psycopg.AsyncCursor, query: str, params: Optional[dict]
    ) -> List[Dict[str, Any]]:
        """Run one Cypher query on ``cursor``; rows as dicts keyed by column name.

        🚨 **NUL is scrubbed here**, at the one boundary. PostgreSQL text cannot
        hold it, and it can arrive on paths that never build a property map: a
        tag list, a filter value, graph metadata, an id. Scrubbing here also
        keeps write and read symmetric: a value stored without its NUL is found
        only by a key that lost the same NUL.
        """
        given = params or {}
        scrubbed = scrub_nul(given)
        if scrubbed is not given:  # scrub_nul returns its input untouched otherwise
            logger.warning("Stripped NUL byte(s) from query parameters before sending")
        columns = return_columns(query)
        await self._execute(
            cursor,
            compose(self._graph_name, query, columns),
            (json.dumps(scrubbed, default=str),),
        )
        if not columns:
            return []
        return [
            {name: agtype.parse(value) for name, value in zip(columns, row)}
            for row in await cursor.fetchall()
        ]

    @asynccontextmanager
    async def _session(self, lock_keys: Iterable[int] = ()) -> AsyncIterator[_Session]:
        """One transaction; yields the runner of its statements.

        With ``lock_keys``, the first statement takes their advisory locks, in
        sorted order (see ``_LOCK_SQL``). The transaction commits when the block
        ends, and that also releases the locks.
        """
        await self._open_pool()
        async with self._pool.connection() as conn:
            async with conn.transaction():
                async with conn.cursor() as cursor:
                    keys = sorted(set(lock_keys))
                    if keys:
                        await self._execute(cursor, _LOCK_SQL, (keys,))
                    yield _Session(self, cursor)

    async def _in_transaction(
        self,
        work: Callable[[_Session], Awaitable[Any]],
        retry: bool = False,
        lock_keys: Iterable[int] = (),
    ) -> Any:
        """Run ``work`` in one transaction; its result.

        With ``retry``, a transaction that lost a race (see ``_lost_a_race``)
        runs again, up to WRITE_ATTEMPTS times. Only idempotent writes ask for it.
        ``lock_keys`` are taken first, as in ``_session``.
        """
        keys = list(lock_keys)
        attempt = 0
        while True:
            attempt += 1
            try:
                async with self._session(keys) as run:
                    return await work(run)
            except psycopg.Error as exc:
                if not retry or attempt >= WRITE_ATTEMPTS or not _lost_a_race(exc):
                    raise
                await _wait_before_retry(attempt, exc)

    async def _transaction(
        self,
        statements: List[Tuple[str, Optional[dict]]],
        retry: bool = False,
        lock_keys: Iterable[int] = (),
    ) -> List[List[Dict[str, Any]]]:
        """Run ``statements`` in one transaction; the rows of each (see ``_in_transaction``)."""

        async def work(run: _Session) -> List[List[Dict[str, Any]]]:
            return [await run(query, params) for query, params in statements]

        return await self._in_transaction(work, retry=retry, lock_keys=lock_keys)

    def _node_lock_keys(self, node_ids: Iterable[str]) -> List[int]:
        return [_lock_key(self._graph_name, "node", str(node_id)) for node_id in node_ids]

    def _edge_lock_keys(self, edges: Iterable[Tuple[str, str, str]]) -> List[int]:
        """Keys of edge identities ``(source, target, relationship name)``.

        The identity, not the two end nodes: every edge of a NodeSet ends at the
        one NodeSet node, so a lock on the ends would make every writer of a set
        wait for every other.
        """
        return [
            _lock_key(self._graph_name, "edge", str(s), str(t), str(rel)) for s, t, rel in edges
        ]

    async def _write(self, query: str, params: Optional[dict] = None) -> List[Dict[str, Any]]:
        """Run one idempotent write, again if it loses a race to another writer."""
        return (await self._transaction([(query, params)], retry=True))[0]

    async def query(self, query: str, params: Optional[dict] = None) -> List[Any]:
        """Run a Cypher query and return its rows as dicts keyed by column name.

        A returned vertex or edge becomes its property map, as neo4j's
        ``Result.data()`` returns it. This is also the shape that cognee's CYPHER
        and NATURAL_LANGUAGE retrievers hand to an LLM. The query must name each
        returned item: ``RETURN *`` is refused, because AGE needs the column list
        before it runs the query.
        """
        return (await self._transaction([(query, params)]))[0]

    async def _sql(self, sql: str, args: Any = None) -> List[Tuple[Any, ...]]:
        """Run one SQL statement outside Cypher (graph and index management)."""
        await self._open_pool()
        async with self._pool.connection() as conn:
            async with conn.cursor() as cursor:
                await self._execute(cursor, sql, args)
                return await cursor.fetchall() if cursor.description else []

    async def close(self) -> None:
        """Close the pool. The next call opens a new pool.

        The contract fixture calls this after every case. ``initialize()`` does
        not run again: the graph, its labels and its indexes stay on the server.
        """
        async with self._pool_lock:
            await self._pool.close()
            self._pool = self._new_pool()
            self._pool_open = False

    async def initialize(self) -> None:
        """Create the graph, its three labels and every index, if absent.

        🚨 **NOTHING ELSE CREATES THESE, AND THE FAILURE IS SILENT.** Without the
        ``id`` index each lookup reads the whole label, and the graph gets slower
        with every node. ``test_plans.py`` asserts the plan of every query.
        Without the index on the ``id`` column of the edge label, each edge SET
        reads the whole edge table. No plan shows that scan, so
        ``test_indexes.py`` asserts the catalog.

        🚨 Labels are created **one statement in each transaction**. Measured:
        3,000 labels created in one transaction failed with ``out of shared
        memory``, because the lock table filled.

        Idempotent, and runs once for each adapter instance: cognee's
        ``get_graph_engine()`` calls it on every engine lookup.
        """
        if self._initialized:
            return
        graph = self._graph_name
        # Created blind, and "already exists" is the normal case. The graph role
        # has no privilege on AGE's catalog tables, so it cannot look first.
        await self._create_if_absent("SELECT ag_catalog.create_graph(%s)", (graph,))
        await self._create_if_absent("SELECT ag_catalog.create_vlabel(%s, %s)", (graph, N))
        await self._create_if_absent(
            "SELECT ag_catalog.create_vlabel(%s, %s)", (graph, METADATA_LABEL)
        )
        await self._create_if_absent("SELECT ag_catalog.create_elabel(%s, %s)", (graph, E))
        for statement in self._index_statements():
            await self._sql(statement)
        self._initialized = True

    async def _create_if_absent(self, sql: str, args: Tuple[str, ...]) -> None:
        """Run one AGE create; a graph or label that exists is not an error.

        Measured on AGE 1.8.0: a repeated ``create_graph``, ``create_vlabel`` or
        ``create_elabel`` fails with SQLSTATE 3F000 (``invalid_schema_name``)
        and the text ``... already exists``.
        """
        try:
            await self._sql(sql, args)
        except errors.InvalidSchemaName as exc:
            if "already exists" not in str(exc):
                raise

    def _index_statements(self) -> List[str]:
        """``CREATE INDEX IF NOT EXISTS`` for every index the queries rely on.

        The expressions are exactly what AGE compiles ``n.<property>`` into, so
        the planner can match them: ``agtype_access_operator(VARIADIC
        ARRAY[properties, '"<property>"'::agtype])`` (``_property``).
        """
        graph = self._graph_name

        def table(label: str) -> str:
            return f'"{graph}"."{label}"'

        statements = []
        for prop, unique in NODE_BTREE_INDEXES:
            statements.append(
                f"CREATE {'UNIQUE ' if unique else ''}INDEX IF NOT EXISTS "
                f'"{N.lower()}_{prop}_btree" ON {table(N)} ({_property(prop)})'
            )
        for prop in NODE_GIN_INDEXES:
            statements.append(
                f'CREATE INDEX IF NOT EXISTS "{N.lower()}_{prop}_gin" '
                f"ON {table(N)} USING gin ({_property(prop)})"
            )
        # The edge identity of cognee, (source, target, relationship name). It is
        # UNIQUE so that two concurrent writers cannot make the same edge twice.
        # 🚨 A relationship name longer than approximately 2.7 kB fails the write.
        statements.append(
            f'CREATE UNIQUE INDEX IF NOT EXISTS "{E.lower()}_identity" '
            f"ON {table(E)} (start_id, end_id, {_property(REL)})"
        )
        statements.append(
            f'CREATE INDEX IF NOT EXISTS "{E.lower()}_{REL}_btree" ON {table(E)} ({_property(REL)})'
        )
        # 🚨 The graphid column ``id``, not a property. AGE 1.8.0 indexes ``id``
        # for a vertex label only (label_commands.c, line 430). A Cypher SET
        # finds its row through a btree index on column 1, else it scans the
        # whole label table (cypher_set.c, lines 717-799). The plan gate cannot
        # see that scan: it is inside the "Cypher Set" node.
        statements.append(
            f'CREATE INDEX IF NOT EXISTS "{E.lower()}_id_btree" ON {table(E)} USING btree (id)'
        )
        for prop in EDGE_GIN_INDEXES:
            statements.append(
                f'CREATE INDEX IF NOT EXISTS "{E.lower()}_{prop}_gin" '
                f"ON {table(E)} USING gin ({_property(prop)})"
            )
        statements.append(
            f'CREATE UNIQUE INDEX IF NOT EXISTS "{METADATA_LABEL.lower()}_id_btree" '
            f"ON {table(METADATA_LABEL)} ({_property('id')})"
        )
        return statements

    # ------------------------------------------------------------------
    # Property serialization
    # ------------------------------------------------------------------

    def serialize_properties(self, properties: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Convert a property dict into the values the store keeps.

        ``weights`` is expanded into ``weight_<name>`` scalars (cognee's
        visualization preprocessor reads that shape), and everything then goes
        through :func:`~.coercion.coerce_properties`.

        🚨 A ``None`` value leaves as an **absent key**, not as a null: AGE
        treats a null as a *delete*.
        """
        flattened: Dict[str, Any] = {}

        for key, value in (properties or {}).items():
            if key == "weights" and isinstance(value, dict):
                flattened.update({f"weight_{name}": weight for name, weight in value.items()})
            else:
                flattened[key] = value

        return coerce_properties(flattened)

    def _node_payload(self, node: Any) -> Tuple[str, Dict[str, Any]]:
        """Return ``(node_id, serialized_properties)`` for one input node.

        Accepts every shape of the interface's ``Union[List[Node],
        List[DataPoint]]``: a DataPoint (or anything with ``model_dump``), the
        ``(id, properties)`` tuple the ``Node`` alias names, or a bare property
        dict. The cognee type goes into the ``type`` property: a DataPoint
        dumps it, and an object without one gets its class name, as the
        FalkorDB adapter gave it as a label. A bare payload declares no type.
        """
        if isinstance(node, (tuple, list)) and len(node) == 2 and isinstance(node[1], dict):
            node_id, raw = str(node[0]), dict(node[1])
            raw.setdefault("id", node_id)
        elif hasattr(node, "model_dump"):
            raw = node.model_dump()
            node_id = str(raw.get("id", getattr(node, "id", "")))
        elif isinstance(node, dict):
            raw = dict(node)
            node_id = str(raw.get("id", ""))
        else:
            raw = dict(vars(node))
            node_id = str(raw.get("id", getattr(node, "id", "")))

        properties = self.serialize_properties(raw)
        properties["id"] = node_id
        if properties.get("belongs_to_set"):
            properties["belongs_to_set"] = _dedupe(properties["belongs_to_set"])
        if not properties.get("type") and not isinstance(node, (dict, tuple, list)):
            properties["type"] = type(node).__name__
        return node_id, properties

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    async def has_node(self, node_id: str) -> bool:
        """Return True when a node with this id exists."""
        rows = await self.query(
            f"MATCH (n:{N}) WHERE n.id = $node_id RETURN count(n) > 0 AS node_exists",
            {"node_id": str(node_id)},
        )
        return bool(rows[0]["node_exists"]) if rows else False

    async def is_empty(self) -> bool:
        """Return True if the graph contains no data nodes.

        Scoped to the data label, so the metadata vertex never makes an empty
        graph read as non-empty. ``LIMIT 1``, not a count: the answer needs one
        row.
        """
        rows = await self.query(f"MATCH (n:{N}) RETURN n.id AS id LIMIT 1")
        return not rows

    async def add_node(
        self, node: Union[DataPoint, str], properties: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add or update a single node: a DataPoint, or a bare id and a property dict."""
        if isinstance(node, str):
            payload = dict(properties or {})
            payload["id"] = node
            await self.add_nodes([payload])
        else:
            await self.add_nodes([node])

    async def add_nodes(
        self,
        nodes: Union[List[Node], List[DataPoint]],
        source_ref_key: Optional[str] = None,
        pipeline_run_id: Optional[str] = None,
    ) -> None:
        """Add or update many nodes, stamping provenance in the same write.

        Three statements in ONE transaction, so no reader sees a half-written node:

        1. Lock each node id (``_LOCK_SQL``), so that a concurrent writer of the
           same nodes commits first.
        2. Create a bare ``{id}`` vertex for each id that is absent.
        3. Update every node of the batch through the unique ``id`` index:
           ``SET n += properties``, the union of ``belongs_to_set``, and the
           provenance fold.

        🚨 The batch is deduplicated and SORTED by id first. A repeated id in
        step 2 fails the statement (the second create of one id). A batch of
        more than LOCK_BATCH nodes is written in transactions of LOCK_BATCH
        nodes, in id order, so that one write cannot fill the lock table. A
        write that still loses a race (a unique violation, or AGE's
        ``Entity failed to be updated``) runs again (``retry``).
        """
        fold_clause = ""
        provenance_params: Dict[str, Any] = {}
        if source_ref_key is not None:
            fold_clause = _provenance_fold_clause("n")
            provenance_params = _provenance_fold_params(source_ref_key, pipeline_run_id)

        # Dedup by node_id within the batch, merging the tags of the copies: the
        # second copy of a node would otherwise recompute the tag union from the
        # pre-SET state, dropping any tag only the first carried.
        deduped: Dict[str, Dict[str, Any]] = {}
        for node in nodes:
            node_id, node_properties = self._node_payload(node)
            existing = deduped.get(node_id)
            if existing:
                merged = _dedupe(
                    list(existing.get("belongs_to_set") or [])
                    + list(node_properties.get("belongs_to_set") or [])
                )
                if merged:
                    node_properties["belongs_to_set"] = merged
            deduped[node_id] = node_properties
        if not deduped:
            return
        batch = [
            {"node_id": node_id, "properties": deduped[node_id]} for node_id in sorted(deduped)
        ]

        create_missing = f"""
            UNWIND $ids AS wanted
            OPTIONAL MATCH (n:{N}) WHERE n.id = wanted
            WITH wanted, n WHERE n IS NULL
            CREATE (:{N} {{id: wanted}})
        """
        update = f"""
            UNWIND $nodes AS node
            MATCH (n:{N}) WHERE n.id = node.node_id
            WITH n, node, node.properties AS props,
                 coalesce(n.belongs_to_set, []) AS existing_tags,
                 coalesce(n.source_ref_keys, []) AS prov_keys_before
            WITH n, props, prov_keys_before,
                 existing_tags
                 + [tag IN coalesce(props.belongs_to_set, [])
                    WHERE NOT tag IN existing_tags] AS merged_belongs_to_set
            SET n += props
            SET n.updated_at = timestamp(), n.belongs_to_set = merged_belongs_to_set
            {fold_clause}
            RETURN n.id AS node_id
        """

        # 🚨 **A folded write takes the attach/remove lock.** The fold stamps the
        # owner key inside the update, which is atomic on its own, but
        # ``attach_*_source_refs`` / ``remove_*_source_refs`` are a read-then-write
        # pair under ``_source_ref_change_lock``. A fold that commits between
        # their read and their write is overwritten. That is what two documents
        # of one cognify run do to an entity they share. Ported from cognee
        # 1.5.4's neo4j and ladybug adapters; the upstream gate is the contract
        # case ``test_concurrent_folded_writes_and_attaches_keep_every_owner``.
        fold_lock = self._source_ref_change_lock if source_ref_key is not None else nullcontext()
        async with fold_lock:
            for part in _in_lock_batches(batch):
                ids = [entry["node_id"] for entry in part]
                await self._transaction(
                    [
                        (create_missing, {"ids": ids}),
                        (update, {"nodes": list(part), **provenance_params}),
                    ],
                    retry=True,
                    lock_keys=self._node_lock_keys(ids),
                )

    async def delete_node(self, node_id: str) -> None:
        """Delete one node and its relationships."""
        await self.delete_nodes([node_id])

    async def delete_nodes(self, node_ids: List[str]) -> None:
        """Delete many nodes and their relationships."""
        if not node_ids:
            return
        await self._write(
            f"""
            UNWIND $node_ids AS wanted
            MATCH (n:{N}) WHERE n.id = wanted
            DETACH DELETE n
            """,
            {"node_ids": sorted({str(node_id) for node_id in node_ids})},
        )

    async def remove_belongs_to_set_tags(
        self,
        tags: List[str],
        node_ids: Optional[List[str]] = None,
    ) -> None:
        """Strip tag names from ``belongs_to_set`` and drop the stale NodeSet edges.

        📌 Not optional: the interface's default is a no-op, and the contract
        suite asserts that the tags are gone.

        Two statements in one transaction. The first strips the property; the
        GIN index on ``belongs_to_set`` serves ``?|`` (any of the tags). The
        second deletes the ``belongs_to_set`` edges to the NodeSets of those
        names, also from a node that never carried the tag in its list.
        """
        if not tags or (node_ids is not None and not node_ids):
            return

        params: Dict[str, Any] = {"tags": list(tags)}
        if node_ids is None:
            strip = f"""
                MATCH (n:{N}) WHERE n.belongs_to_set ?| $tags
                SET n.belongs_to_set = [x IN n.belongs_to_set WHERE NOT x IN $tags]
            """
            scope = ""
        else:
            params["node_ids"] = sorted({str(node_id) for node_id in node_ids})
            strip = f"""
                UNWIND $node_ids AS wanted
                MATCH (n:{N}) WHERE n.id = wanted AND n.belongs_to_set ?| $tags
                SET n.belongs_to_set = [x IN n.belongs_to_set WHERE NOT x IN $tags]
            """
            scope = " AND n.id IN $node_ids"
        prune = f"""
            MATCH (ns:{N}) WHERE ns.type = 'NodeSet' AND ns.name IN $tags
            MATCH (n:{N})-[r:{E}]->(ns) WHERE r.{REL} = 'belongs_to_set'{scope}
            DELETE r
        """
        await self._transaction([(strip, params), (prune, params)], retry=True)

    async def get_node(self, node_id: str) -> Optional[NodeData]:
        """Retrieve one node's properties, or None."""
        rows = await self.query(
            f"MATCH (n:{N}) WHERE n.id = $node_id RETURN properties(n) AS properties",
            {"node_id": str(node_id)},
        )
        return _strip_provenance(rows[0]["properties"]) if rows else None

    async def get_nodes(self, node_ids: List[str]) -> List[NodeData]:
        """Retrieve many nodes' properties. Missing ids are simply absent."""
        if not node_ids:
            return []
        rows = await self._nodes_by_id([str(node_id) for node_id in node_ids])
        return [_strip_provenance(row["properties"]) for row in rows]

    async def _nodes_by_id(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        """``properties(n)`` rows of the nodes with these ids, each id once."""
        if not node_ids:
            return []
        return await self.query(
            f"""
            UNWIND $ids AS wanted
            MATCH (n:{N}) WHERE n.id = wanted
            RETURN properties(n) AS properties
            """,
            {"ids": _dedupe(node_ids)},
        )

    async def get_neighbors(self, node_id: str) -> List[NodeData]:
        """Every node directly connected to this one, in either direction."""
        rows = await self.query(
            f"""
            MATCH (n:{N})-[:{E}]->(m:{N}) WHERE n.id = $node_id
            RETURN properties(m) AS properties
            UNION
            MATCH (n:{N})<-[:{E}]-(m:{N}) WHERE n.id = $node_id
            RETURN properties(m) AS properties
            """,
            {"node_id": str(node_id)},
        )
        return [_strip_provenance(row["properties"]) for row in rows]

    async def get_connections(
        self, node_id: Union[str, UUID]
    ) -> List[Tuple[NodeData, Dict[str, Any], NodeData]]:
        """Every (predecessor, relationship, node) and (node, relationship, successor) triple."""
        rows = await self.query(
            f"""
            MATCH (n:{N})<-[r:{E}]-(m:{N}) WHERE n.id = $node_id
            RETURN properties(m) AS source, r.{REL} AS rel, properties(n) AS target
            UNION ALL
            MATCH (n:{N})-[r:{E}]->(m:{N}) WHERE n.id = $node_id
            RETURN properties(n) AS source, r.{REL} AS rel, properties(m) AS target
            """,
            {"node_id": str(node_id)},
        )
        return [
            (
                _strip_provenance(row["source"]),
                {"relationship_name": row["rel"]},
                _strip_provenance(row["target"]),
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------

    async def add_edge(
        self,
        source_id: str,
        target_id: str,
        relationship_name: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create or update one edge."""
        await self.add_edges([(source_id, target_id, relationship_name, properties or {})])

    async def add_edges(
        self,
        edges: Union[List[EdgeData], List[Tuple[str, str, str, Optional[Dict[str, Any]]]]],
        source_ref_key: Optional[str] = None,
        pipeline_run_id: Optional[str] = None,
    ) -> None:
        """Add or update many edges, stamping provenance in the same write.

        The relationship name is the ``relationship_name`` property of the one
        edge label, so one statement writes every relationship name of the
        batch. The ``MERGE`` finds an existing edge through the ``end_id`` index
        of its target. Properties are SET after the MERGE, so a re-cognify
        updates an existing edge, and ``created_at`` keeps the first write's
        timestamp.

        The batch is deduplicated by edge identity (later properties win, as
        two SETs in sequence would), sorted, locked by edge identity and
        written in transactions of LOCK_BATCH edges, for the reasons in
        ``add_nodes``. An edge whose endpoint is absent is not written.
        """
        if not edges:
            return

        fold_clause = ""
        provenance_params: Dict[str, Any] = {}
        if source_ref_key is not None:
            fold_clause = _provenance_fold_clause("rel")
            provenance_params = _provenance_fold_params(source_ref_key, pipeline_run_id)

        deduped: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for edge in edges:
            source, target, relationship_name = str(edge[0]), str(edge[1]), str(edge[2])
            edge_properties = edge[3] if len(edge) > 3 and edge[3] else {}
            identity = (source, target, scrub_nul(relationship_name))
            properties = self.serialize_properties(
                {**edge_properties, "source_node_id": source, "target_node_id": target}
            )
            # The identity name, never another value from the property dict.
            properties[REL] = identity[2]
            deduped[identity] = {**deduped.get(identity, {}), **properties}
        batch = [
            {**self._edge_identity_row(EdgeIdentity(s, t, rel)), "properties": deduped[(s, t, rel)]}
            for s, t, rel in sorted(deduped)
        ]

        query = f"""
            UNWIND $edges AS edge
            MATCH (source:{N}) WHERE source.id = edge.s
            MATCH (target:{N}) WHERE target.id = edge.t
            MERGE (source)-[rel:{E} {{{REL}: edge.rel}}]->(target)
            WITH rel, edge, edge.properties AS props, coalesce(rel.source_ref_keys, []) AS prov_keys_before
            SET rel += props
            SET rel.updated_at = timestamp(), rel.created_at = coalesce(rel.created_at, timestamp())
            {fold_clause}
            RETURN rel.{REL} AS relationship_name
        """
        # Same lock as ``add_nodes``, for the same race with the edge attach/remove.
        fold_lock = self._source_ref_change_lock if source_ref_key is not None else nullcontext()
        async with fold_lock:
            for part in _in_lock_batches(batch):
                await self._transaction(
                    [(query, {"edges": list(part), **provenance_params})],
                    retry=True,
                    lock_keys=self._edge_lock_keys((e["s"], e["t"], e["rel"]) for e in part),
                )

    async def has_edge(self, source_id: str, target_id: str, relationship_name: str) -> bool:
        """Return True when this exact edge exists."""
        found = await self.has_edges([(source_id, target_id, relationship_name, {})])
        return bool(found)

    async def has_edges(self, edges: List[EdgeData]) -> List[EdgeData]:
        """Return the subset of ``edges`` that exists, as ladybug and the interface do."""
        if not edges:
            return []
        rows = [
            self._edge_identity_row(
                EdgeIdentity(
                    source_id=str(edge[0]), target_id=str(edge[1]), relationship_name=str(edge[2])
                )
            )
            for edge in edges
        ]
        ids = await self._in_transaction(lambda run: run.edge_ids(rows))
        return [edge for edge, gid in zip(edges, ids) if gid is not None]

    async def get_edges(self, node_id: str) -> List[EdgeData]:
        """Every edge touching this node, as ``(node, other, {relationship_name})``."""
        rows = await self.query(
            f"""
            MATCH (n:{N})-[r:{E}]->(m:{N}) WHERE n.id = $node_id
            RETURN n.id AS node_id, m.id AS other_id, r.{REL} AS relationship_name
            UNION ALL
            MATCH (n:{N})<-[r:{E}]-(m:{N}) WHERE n.id = $node_id
            RETURN n.id AS node_id, m.id AS other_id, r.{REL} AS relationship_name
            """,
            {"node_id": str(node_id)},
        )
        return [
            (row["node_id"], row["other_id"], {"relationship_name": row["relationship_name"]})
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Graph provenance
    #
    # The four provenance fields live in native list properties. attach/remove
    # do a per-artifact read-modify-write under a lock (delete and rollback are
    # maintenance paths, not hot paths). A lookup by one key reads a GIN index
    # (``@>``). Every read normalizes a missing property to []. Edges are
    # addressed by (source id, target id, relationship name).
    # ------------------------------------------------------------------

    @staticmethod
    def _node_identity_row(node_id: str) -> dict:
        return {"id": node_id}

    @staticmethod
    def _edge_identity_row(edge: EdgeIdentity) -> dict:
        return {"s": edge.source_id, "t": edge.target_id, "rel": scrub_nul(edge.relationship_name)}

    @staticmethod
    def _indexed_fields_from_properties(properties: Dict[str, Any]) -> List[str]:
        """Read ``metadata.index_fields`` from a node's stored properties.

        ``metadata`` is a map, so the coercion layer stores it JSON-encoded.
        """
        metadata = properties.get("metadata")
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except (json.JSONDecodeError, TypeError):
                metadata = {}
        if isinstance(metadata, dict):
            return list(metadata.get("index_fields") or [])
        return []

    async def _read_node_provenance(
        self, node_ids: List[str], run: _Session
    ) -> Dict[str, Tuple[List[str], List[str]]]:
        """Return ``{node_id: (source_ref_keys, source_run_refs)}`` for existing nodes."""
        rows = await run(
            f"""
            UNWIND $ids AS wanted
            MATCH (n:{N}) WHERE n.id = wanted
            RETURN n.id AS id,
                   coalesce(n.source_ref_keys, []) AS keys,
                   coalesce(n.source_run_refs, []) AS run_refs
            """,
            {"ids": _dedupe(str(node_id) for node_id in node_ids)},
        )
        return {row["id"]: (_prov_list(row["keys"]), _prov_list(row["run_refs"])) for row in rows}

    async def _write_node_provenance(self, batch: List[dict], run: _Session) -> None:
        if not batch:
            return
        await run(
            f"""
            UNWIND $batch AS row
            MATCH (n:{N}) WHERE n.id = row.id
            SET n.source_ref_keys = row.refs,
                n.source_dataset_ids = row.datasets,
                n.source_run_ids = row.runs,
                n.source_run_refs = row.run_refs
            """,
            {"batch": sorted(batch, key=lambda row: row["id"])},
        )

    async def _read_edge_provenance(
        self, edges: List[EdgeIdentity], run: _Session
    ) -> Dict[EdgeIdentity, Tuple[List[str], List[str]]]:
        """Return ``{edge: (source_ref_keys, source_run_refs)}`` for existing edges."""
        rows = await run.on_edges(
            [self._edge_identity_row(edge) for edge in edges],
            f"""
            RETURN a.id AS s, b.id AS t, r.{REL} AS rel,
                   coalesce(r.source_ref_keys, []) AS keys,
                   coalesce(r.source_run_refs, []) AS run_refs
            """,
        )
        return {
            EdgeIdentity(
                source_id=row["s"], target_id=row["t"], relationship_name=row["rel"]
            ): (_prov_list(row["keys"]), _prov_list(row["run_refs"]))
            for row in rows
        }

    async def _write_edge_provenance(self, batch: List[dict], run: _Session) -> None:
        if not batch:
            return
        await run.on_edges(
            sorted(batch, key=lambda row: (row["s"], row["t"], row["rel"])),
            """
            SET r.source_ref_keys = e.refs,
                r.source_dataset_ids = e.datasets,
                r.source_run_ids = e.runs,
                r.source_run_refs = e.run_refs
            """,
        )

    async def _apply_source_ref_change(
        self,
        artifacts,
        read_provenance,
        write_provenance,
        identity_row,
        transition,
        lock_keys,
    ) -> None:
        """Read each artifact's provenance, apply a pure transition, write it back.

        Shared by attach/remove for both nodes and edges. The read and the
        write run in ONE transaction, after the advisory locks of the artifacts
        (``lock_keys``). So a writer in another process, which takes the same
        locks, cannot commit between the read and the write, and the
        transition always sees the current values. cognify calls this path
        for every chunk-scoped owner (``add_data_points``).

        The asyncio lock serializes this read-modify-write with the folded
        ``add_nodes`` / ``add_edges`` of the same adapter, as cognee's own
        adapters do. A write that still loses a race (a writer without the
        locks, such as ``remove_belongs_to_set_tags``) runs again, read and all.
        """
        if not artifacts:
            return
        async with self._source_ref_change_lock:
            for part in _in_lock_batches(_dedupe(artifacts)):
                for attempt in range(1, WRITE_ATTEMPTS + 1):
                    try:
                        async with self._session(lock_keys(part)) as run:
                            current = await read_provenance(list(part), run)
                            batch = []
                            for identity, (keys, run_refs) in current.items():
                                columns = transition(keys, run_refs)
                                batch.append(
                                    {
                                        **identity_row(identity),
                                        "refs": columns.source_ref_keys,
                                        "datasets": columns.source_dataset_ids,
                                        "runs": columns.source_run_ids,
                                        "run_refs": columns.source_run_refs,
                                    }
                                )
                            await write_provenance(batch, run)
                        break
                    except psycopg.Error as exc:
                        if attempt >= WRITE_ATTEMPTS or not _lost_a_race(exc):
                            raise
                        await _wait_before_retry(attempt, exc)

    def _node_artifact_lock_keys(self, node_ids: Sequence[Any]) -> List[int]:
        return self._node_lock_keys(str(node_id) for node_id in node_ids)

    def _edge_artifact_lock_keys(self, edges: Sequence[EdgeIdentity]) -> List[int]:
        rows = (self._edge_identity_row(edge) for edge in edges)
        return self._edge_lock_keys((row["s"], row["t"], row["rel"]) for row in rows)

    async def attach_node_source_refs(
        self,
        node_ids: list[str],
        source_ref_keys: list[str],
        pipeline_run_id: str | None = None,
    ) -> None:
        if not source_ref_keys:
            return
        add_keys = list(source_ref_keys)
        await self._apply_source_ref_change(
            node_ids,
            self._read_node_provenance,
            self._write_node_provenance,
            self._node_identity_row,
            lambda keys, run_refs: provenance_after_attach(
                keys, run_refs, add_keys, pipeline_run_id
            ),
            self._node_artifact_lock_keys,
        )

    async def attach_edge_source_refs(
        self,
        edges: list[EdgeIdentity],
        source_ref_keys: list[str],
        pipeline_run_id: str | None = None,
    ) -> None:
        if not source_ref_keys:
            return
        add_keys = list(source_ref_keys)
        await self._apply_source_ref_change(
            edges,
            self._read_edge_provenance,
            self._write_edge_provenance,
            self._edge_identity_row,
            lambda keys, run_refs: provenance_after_attach(
                keys, run_refs, add_keys, pipeline_run_id
            ),
            self._edge_artifact_lock_keys,
        )

    async def remove_node_source_refs(
        self,
        node_ids: list[str],
        source_ref_keys: list[str],
    ) -> None:
        if not source_ref_keys:
            return
        remove_keys = list(source_ref_keys)
        await self._apply_source_ref_change(
            node_ids,
            self._read_node_provenance,
            self._write_node_provenance,
            self._node_identity_row,
            lambda keys, run_refs: provenance_after_remove(keys, run_refs, remove_keys),
            self._node_artifact_lock_keys,
        )

    async def remove_edge_source_refs(
        self,
        edges: list[EdgeIdentity],
        source_ref_keys: list[str],
    ) -> None:
        if not source_ref_keys:
            return
        remove_keys = list(source_ref_keys)
        await self._apply_source_ref_change(
            edges,
            self._read_edge_provenance,
            self._write_edge_provenance,
            self._edge_identity_row,
            lambda keys, run_refs: provenance_after_remove(keys, run_refs, remove_keys),
            self._edge_artifact_lock_keys,
        )

    async def delete_edge_triples(self, edges: list[EdgeIdentity]) -> None:
        """Delete exactly these relationships, keeping both endpoints."""
        if not edges:
            return
        rows = [self._edge_identity_row(edge) for edge in edges]
        # DELETE r, never DETACH DELETE: the endpoints are other datasets' nodes.
        await self._in_transaction(lambda run: run.on_edges(rows, "DELETE r"), retry=True)

    async def get_node_delete_data(self, node_ids: list[str]) -> dict[str, NodeDeleteData]:
        if not node_ids:
            return {}
        rows = await self.query(
            f"""
            UNWIND $ids AS wanted
            MATCH (n:{N}) WHERE n.id = wanted
            RETURN n.id AS id, properties(n) AS properties,
                   coalesce(n.source_ref_keys, []) AS srk,
                   coalesce(n.source_dataset_ids, []) AS sdi,
                   coalesce(n.source_run_ids, []) AS sri,
                   coalesce(n.source_run_refs, []) AS srr
            """,
            {"ids": _dedupe(str(node_id) for node_id in node_ids)},
        )
        result: dict[str, NodeDeleteData] = {}
        for row in rows:
            properties = row["properties"] or {}
            result[row["id"]] = NodeDeleteData(
                node_id=row["id"],
                node_type=properties.get("type") or "",
                indexed_fields=self._indexed_fields_from_properties(properties),
                node_properties=_strip_provenance(properties),
                source_ref_keys=_prov_list(row["srk"]),
                source_dataset_ids=_prov_list(row["sdi"]),
                source_run_ids=_prov_list(row["sri"]),
                source_run_refs=_prov_list(row["srr"]),
            )
        return result

    async def get_edge_delete_data(
        self, edges: list[EdgeIdentity]
    ) -> dict[EdgeIdentity, EdgeDeleteData]:
        if not edges:
            return {}
        wanted = [self._edge_identity_row(edge) for edge in edges]
        tail = f"""
            RETURN a.id AS s, b.id AS t, r.{REL} AS rel, properties(r) AS properties,
                   coalesce(r.source_ref_keys, []) AS srk,
                   coalesce(r.source_dataset_ids, []) AS sdi,
                   coalesce(r.source_run_ids, []) AS sri,
                   coalesce(r.source_run_refs, []) AS srr
            """
        rows = await self._in_transaction(lambda run: run.on_edges(wanted, tail))
        # Lazy import: this helper lives in cognee's modules layer, whose package
        # __init__ reaches the graph engine. At delete time the cycle is closed.
        from cognee.modules.graph.utils.prepare_edges_for_storage import get_edge_retrieval_text

        result: dict[EdgeIdentity, EdgeDeleteData] = {}
        for row in rows:
            edge = EdgeIdentity(
                source_id=row["s"], target_id=row["t"], relationship_name=row["rel"]
            )
            properties = row["properties"] or {}
            result[edge] = EdgeDeleteData(
                edge=edge,
                # Stored edge_text wins; fall back to the relationship name.
                edge_text=get_edge_retrieval_text(
                    properties.get("edge_text"), edge.relationship_name
                ),
                edge_properties=_strip_provenance(properties),
                source_ref_keys=_prov_list(row["srk"]),
                source_dataset_ids=_prov_list(row["sdi"]),
                source_run_ids=_prov_list(row["sri"]),
                source_run_refs=_prov_list(row["srr"]),
            )
        return result

    async def find_nodes_by_source_ref(self, source_ref_key: str) -> list[str]:
        rows = await self.query(
            f"MATCH (n:{N}) WHERE n.source_ref_keys @> [$token] RETURN n.id AS id",
            {"token": source_ref_key},
        )
        return [row["id"] for row in rows]

    async def find_edges_by_source_ref(self, source_ref_key: str) -> list[EdgeIdentity]:
        rows = await self.query(
            f"""
            MATCH (a:{N})-[r:{E}]->(b:{N}) WHERE r.source_ref_keys @> [$token]
            RETURN a.id AS s, b.id AS t, r.{REL} AS rel
            """,
            {"token": source_ref_key},
        )
        return [
            EdgeIdentity(source_id=row["s"], target_id=row["t"], relationship_name=row["rel"])
            for row in rows
        ]

    async def find_node_source_refs_by_dataset(self, dataset_id: str) -> dict[str, list[str]]:
        rows = await self.query(
            f"""
            MATCH (n:{N}) WHERE n.source_dataset_ids @> [$token]
            RETURN n.id AS id, coalesce(n.source_ref_keys, []) AS keys
            """,
            {"token": dataset_id},
        )
        result: dict[str, list[str]] = {}
        for row in rows:
            owned = [
                key
                for key in _prov_list(row["keys"])
                if str(get_dataset_id_from_source_ref_key(key)) == dataset_id
            ]
            if owned:
                result[row["id"]] = owned
        return result

    async def find_edge_source_refs_by_dataset(
        self, dataset_id: str
    ) -> dict[EdgeIdentity, list[str]]:
        rows = await self.query(
            f"""
            MATCH (a:{N})-[r:{E}]->(b:{N}) WHERE r.source_dataset_ids @> [$token]
            RETURN a.id AS s, b.id AS t, r.{REL} AS rel, coalesce(r.source_ref_keys, []) AS keys
            """,
            {"token": dataset_id},
        )
        result: dict[EdgeIdentity, list[str]] = {}
        for row in rows:
            owned = [
                key
                for key in _prov_list(row["keys"])
                if str(get_dataset_id_from_source_ref_key(key)) == dataset_id
            ]
            if owned:
                result[
                    EdgeIdentity(
                        source_id=row["s"], target_id=row["t"], relationship_name=row["rel"]
                    )
                ] = owned
        return result

    # --- per-document lookups: NOT on GraphDBInterface -----------------------
    #
    # cognee's ``UnifiedStoreEngine.delete_by_document`` resolves one document's
    # refs with ``find_*_source_refs_by_dataset`` (every node and edge of the
    # dataset, with its whole key list) and filters to the data id in Python.
    # The homelab cognee patch calls these instead when the adapter has them
    # (``hasattr``), so their result must EQUAL that by_dataset-then-filter
    # computation. The filter runs server-side, as a superset
    # (``CONTAINS $data_id``), and ``_document_refs`` makes it exact.

    async def find_node_source_refs_by_document(
        self, dataset_id: str, data_id: str
    ) -> dict[str, list[str]]:
        """``{node_id: keys}`` for the refs one document of one dataset owns."""
        if not data_id:
            return {}
        rows = await self.query(
            f"""
            MATCH (n:{N}) WHERE n.source_dataset_ids @> [$dataset_id]
            WITH n, [key IN n.source_ref_keys WHERE key CONTAINS $data_id] AS keys
            WHERE size(keys) > 0
            RETURN n.id AS id, keys
            """,
            {"dataset_id": dataset_id, "data_id": data_id},
        )
        result: dict[str, list[str]] = {}
        for row in rows:
            owned = _document_refs(_prov_list(row["keys"]), dataset_id, data_id)
            if owned:
                result[row["id"]] = owned
        return result

    async def find_edge_source_refs_by_document(
        self, dataset_id: str, data_id: str
    ) -> dict[EdgeIdentity, list[str]]:
        """``{edge: keys}`` for the refs one document of one dataset owns.

        Not a read of every edge of the dataset. Two index-backed reads, unioned
        by ``EdgeIdentity``:

        1. **Anchored**: every edge incident to a node this document owns
           (``find_node_source_refs_by_document``), in either direction.
        2. **Chunk sweep**: every ``DocumentChunk -> DocumentChunk`` edge of the
           dataset.

        🚨 **Complete only under an invariant of cognee's write path**: an edge
        that carries document D's ref has an endpoint that also carries a D ref,
        OR is chunk->chunk. The FalkorDB adapter's docstring lists the writers
        (cognee 1.5.4, re-verified up to 1.6.7). ``create_chunk_associations`` is
        the exception, and the reason for the sweep: it can stamp D's ref on a
        chunk->chunk edge between two OTHER documents' chunks.

        ⚠ An edge that carries D's ref, whose endpoints own no D ref, and that
        is not chunk->chunk, is NOT returned. No cognee write path makes one. A
        miss leaves a stale ref (a leak, never an over-delete), and
        ``delete_by_dataset`` still removes it.

        🚨 **The anchored read has no dataset condition that an index serves**
        (homelab#332, D3). With ``r.source_dataset_ids @> [$dataset_id]`` the
        planner started from the GIN index (818 rows estimated, 651,981
        actual) and read every edge of the dataset: 138 s for a document of 22
        nodes. The dataset is checked after the anchor, by ``$dataset_id IN``
        (which no index serves) and exactly by ``_document_refs``. The anchors
        go in slices of LOCK_BATCH, so one statement of a large document stays
        short.
        """
        if not data_id:
            return {}
        params = {"dataset_id": dataset_id, "data_id": data_id}
        node_ids = sorted(await self.find_node_source_refs_by_document(dataset_id, data_id))
        rows = []
        for anchors in _slices(node_ids, ANCHOR_BATCH):
            rows += await self.query(
                _anchored_edges_query(
                    filters=(
                        "$dataset_id IN r.source_dataset_ids AND "
                        "size([key IN r.source_ref_keys WHERE key CONTAINS $data_id]) > 0"
                    ),
                    returns=f"""s.id AS s, t.id AS t, r.{REL} AS rel,
                        [key IN r.source_ref_keys WHERE key CONTAINS $data_id] AS keys""",
                ),
                {**params, "anchor_ids": list(anchors)},
            )
        rows += await self.query(
            f"""
            MATCH (a:{N})-[r:{E}]->(b:{N})
            WHERE r.source_dataset_ids @> [$dataset_id]
              AND a.type = $chunk_type AND b.type = $chunk_type
            WITH a, r, b, [key IN r.source_ref_keys WHERE key CONTAINS $data_id] AS keys
            WHERE size(keys) > 0
            RETURN a.id AS s, b.id AS t, r.{REL} AS rel, keys
            """,
            {**params, "chunk_type": CHUNK_TYPE},
        )
        result: dict[EdgeIdentity, list[str]] = {}
        for row in rows:
            owned = _document_refs(_prov_list(row["keys"]), dataset_id, data_id)
            if owned:
                result[
                    EdgeIdentity(
                        source_id=row["s"], target_id=row["t"], relationship_name=row["rel"]
                    )
                ] = owned
        return result

    async def find_node_source_refs_by_pipeline_run(
        self, pipeline_run_id: str
    ) -> dict[str, list[str]]:
        rows = await self.query(
            f"""
            MATCH (n:{N}) WHERE n.source_run_ids @> [$token]
            RETURN n.id AS id, coalesce(n.source_run_refs, []) AS run_refs
            """,
            {"token": pipeline_run_id},
        )
        result: dict[str, list[str]] = {}
        for row in rows:
            contributed = [
                get_source_ref_key_from_source_run_ref(ref)
                for ref in _prov_list(row["run_refs"])
                if str(get_pipeline_run_id_from_source_run_ref(ref)) == pipeline_run_id
            ]
            if contributed:
                result[row["id"]] = contributed
        return result

    async def find_edge_source_refs_by_pipeline_run(
        self, pipeline_run_id: str
    ) -> dict[EdgeIdentity, list[str]]:
        rows = await self.query(
            f"""
            MATCH (a:{N})-[r:{E}]->(b:{N}) WHERE r.source_run_ids @> [$token]
            RETURN a.id AS s, b.id AS t, r.{REL} AS rel,
                   coalesce(r.source_run_refs, []) AS run_refs
            """,
            {"token": pipeline_run_id},
        )
        result: dict[EdgeIdentity, list[str]] = {}
        for row in rows:
            contributed = [
                get_source_ref_key_from_source_run_ref(ref)
                for ref in _prov_list(row["run_refs"])
                if str(get_pipeline_run_id_from_source_run_ref(ref)) == pipeline_run_id
            ]
            if contributed:
                result[
                    EdgeIdentity(
                        source_id=row["s"], target_id=row["t"], relationship_name=row["rel"]
                    )
                ] = contributed
        return result

    async def set_graph_metadata(self, metadata: dict[str, str]) -> None:
        """Merge graph-level metadata onto the one metadata vertex.

        🚨 The vertex has its own label, never the data label, so ``is_empty()``
        and ``get_graph_data()`` never see it as data.
        """
        if not metadata:
            return
        params = {
            "marker_id": METADATA_NODE_ID,
            "metadata": {str(key): str(value) for key, value in metadata.items()},
        }
        # ⚠ The UNWIND is load-bearing. Measured on AGE 1.8.0: a query that
        # STARTS with an OPTIONAL MATCH that matches nothing returns no row (not
        # one row of nulls), so the CREATE after it never ran.
        await self._transaction(
            [
                (
                    f"""
                    UNWIND [$marker_id] AS wanted
                    OPTIONAL MATCH (m:{METADATA_LABEL}) WHERE m.id = wanted
                    WITH wanted, m WHERE m IS NULL
                    CREATE (:{METADATA_LABEL} {{id: wanted}})
                    """,
                    params,
                ),
                (
                    f"""
                    MATCH (m:{METADATA_LABEL}) WHERE m.id = $marker_id
                    WITH m, $metadata AS metadata
                    SET m += metadata
                    """,
                    params,
                ),
            ],
            retry=True,
        )

    async def get_graph_metadata(self) -> dict[str, str]:
        rows = await self.query(
            f"""
            MATCH (m:{METADATA_LABEL}) WHERE m.id = $marker_id
            RETURN properties(m) AS properties
            """,
            {"marker_id": METADATA_NODE_ID},
        )
        if not rows:
            return {}
        return {
            key: value for key, value in (rows[0]["properties"] or {}).items() if key != "id"
        }

    # ------------------------------------------------------------------
    # Truth state — ported from cognee's ladybug adapter, the only in-core
    # adapter that implements it.
    # ------------------------------------------------------------------

    async def get_node_truth_state(self, node_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Return ``{node_id: {"truth_alignment": [...], "truth_epoch": int|None}}``."""
        valid_ids = [str(node_id) for node_id in node_ids or [] if isinstance(node_id, str) and node_id]
        if not valid_ids:
            return {}
        rows = await self.query(
            f"""
            UNWIND $ids AS wanted
            MATCH (n:{N}) WHERE n.id = wanted
            RETURN n.id AS id, n.truth_alignment AS alignment, n.truth_epoch AS epoch
            """,
            {"ids": _dedupe(valid_ids)},
        )
        result: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            alignment = row["alignment"]
            try:
                truth_epoch = int(row["epoch"]) if row["epoch"] is not None else None
            except (TypeError, ValueError):
                truth_epoch = None
            result[row["id"]] = {
                "truth_alignment": list(alignment) if isinstance(alignment, (list, tuple)) else [],
                "truth_epoch": truth_epoch,
            }
        return result

    async def set_node_truth_state(
        self, node_truth_state: Dict[str, Dict[str, Any]]
    ) -> Dict[str, bool]:
        """Persist truth state per node; returns ``{node_id: updated}``.

        ⚠ A ``truth_epoch`` of ``None`` means "this call carries no epoch", not
        "clear the stored one", so the batch is split and the null is never
        written: ``SET n.truth_epoch = null`` would **delete** the property.
        """
        if not node_truth_state:
            return {}
        node_ids = list(node_truth_state.keys())

        items = []
        for node_id, state in node_truth_state.items():
            if not isinstance(node_id, str) or not node_id:
                continue
            epoch = (state or {}).get("truth_epoch")
            items.append(
                {
                    "node_id": node_id,
                    "alignment": list((state or {}).get("truth_alignment") or []),
                    "epoch": int(epoch) if epoch is not None else None,
                }
            )
        if not items:
            return {node_id: False for node_id in node_ids}

        updated: Set[str] = set()
        with_epoch = [item for item in items if item["epoch"] is not None]
        without_epoch = [item for item in items if item["epoch"] is None]
        set_with_epoch = f"""
            UNWIND $items AS item
            MATCH (n:{N}) WHERE n.id = item.node_id
            SET n.truth_alignment = item.alignment,
                n.truth_epoch = item.epoch,
                n.updated_at = timestamp()
            RETURN n.id AS node_id
        """
        set_without_epoch = f"""
            UNWIND $items AS item
            MATCH (n:{N}) WHERE n.id = item.node_id
            SET n.truth_alignment = item.alignment,
                n.updated_at = timestamp()
            RETURN n.id AS node_id
        """

        # The node locks of add_nodes, for the reason in ``_LOCK_SQL``.
        for query, group in ((set_with_epoch, with_epoch), (set_without_epoch, without_epoch)):
            for part in _in_lock_batches(group):
                [rows] = await self._transaction(
                    [(query, {"items": list(part)})],
                    retry=True,
                    lock_keys=self._node_lock_keys(item["node_id"] for item in part),
                )
                updated |= {row["node_id"] for row in rows}

        return {node_id: node_id in updated for node_id in node_ids}

    # ------------------------------------------------------------------
    # Whole-graph and subgraph reads
    # ------------------------------------------------------------------

    async def delete_graph(self) -> None:
        """Remove every vertex and edge, and keep the graph, its labels and indexes.

        📌 Not ``drop_graph``: that removes the indexes with the data, and only
        ``initialize()`` makes them again.
        """
        await self._write("MATCH (n) DETACH DELETE n")

    async def get_graph_data(self) -> Tuple[List[Node], List[EdgeData]]:
        """Every data node and edge, in the ``(id, properties)`` / 4-tuple shape."""
        node_rows = await self.query(f"MATCH (n:{N}) RETURN properties(n) AS properties")
        edge_rows = await self.query(
            f"""
            MATCH (n:{N})-[r:{E}]->(m:{N})
            RETURN n.id AS source, m.id AS target, r.{REL} AS type, properties(r) AS properties
            """
        )
        nodes = _node_tuples(node_rows)
        edges = [_edge_tuple(row) for row in edge_rows]
        logger.info("Retrieved %d nodes and %d edges", len(nodes), len(edges))
        return (nodes, edges)

    async def get_id_filtered_graph_data(
        self, target_ids: list[str]
    ) -> tuple[list[Node], list[EdgeData]]:
        """The edges that touch ``target_ids``, and the endpoints of those edges.

        Not on ``GraphDBInterface``. ``CogneeGraph._get_full_or_id_filtered_graph``
        calls it when the adapter class has it; without it, GRAPH_COMPLETION search
        reads the full graph through ``get_graph_data``.

        The semantics are those of cognee 1.6.1's neo4j and ladybug adapters: an
        edge is returned when one endpoint id is a target, and a node is returned
        only as an endpoint of such an edge. If no target has an edge, the result
        is empty, and cognee falls back to ``get_graph_data``.
        """
        if not target_ids:
            logger.warning("No target IDs provided for ID-filtered graph retrieval.")
            return [], []

        edge_rows = await self.query(
            _anchored_edges_query(),
            {"anchor_ids": _dedupe(str(target_id) for target_id in target_ids)},
        )
        if not edge_rows:
            return [], []

        endpoint_ids = _dedupe(
            endpoint for row in edge_rows for endpoint in (row["source"], row["target"])
        )
        nodes = _node_tuples(await self._nodes_by_id(endpoint_ids))
        edges = [_edge_tuple(row) for row in edge_rows]
        logger.info("ID-filtered retrieval: %d nodes and %d edges", len(nodes), len(edges))
        return (nodes, edges)

    async def get_neighborhood(
        self,
        node_ids: List[str],
        depth: int = 1,
        edge_types: Optional[List[str]] = None,
    ) -> Tuple[List[Node], List[EdgeData]]:
        """The k-hop subgraph around a set of seed nodes, in ``get_graph_data`` shape.

        ``edge_types`` selects the paths to the neighbours. The returned edges are
        every edge between the collected nodes, of any type, as the in-core
        adapters return them.

        🚨 cognee 1.6.6 calls this on every RAG and CHUNKS search
        (``conflict_context.get_chunk_conflicts``: the hits as seeds,
        ``edge_types=["conflict_cites"]``). The traversal is a breadth-first
        search, one index-backed read for each hop, so its cost follows the
        edges it reaches and not the size of the graph.
        """
        if not node_ids:
            logger.warning("No node IDs provided for neighborhood retrieval.")
            return [], []

        seed_ids = _dedupe(str(node_id) for node_id in node_ids)
        hops = max(1, int(depth))
        wanted_types = _dedupe(scrub_nul(name) for name in edge_types or [])
        type_filter = f"r.{REL} IN $edge_types" if wanted_types else ""

        reached = set(seed_ids)
        frontier = seed_ids
        for _ in range(hops):
            rows = await self.query(
                _anchored_edges_query(filters=type_filter, returns="s.id AS source, t.id AS target"),
                {"anchor_ids": sorted(frontier), "edge_types": wanted_types},
            )
            found = {endpoint for row in rows for endpoint in (row["source"], row["target"])}
            frontier = sorted(found - reached)
            reached |= found
            if not frontier:
                break

        all_ids = sorted(reached)
        nodes = _node_tuples(await self._nodes_by_id(all_ids))
        edges = [_edge_tuple(row) for row in await self._induced_edge_rows(all_ids)]
        return (nodes, edges)

    async def _induced_edge_rows(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        """Every edge between two nodes of ``node_ids``: source, target, type, properties.

        Every outgoing edge of each node, through the ``start_id`` index; the
        target is a set lookup in Python.

        🚨 Not ``AND m.id IN $ids`` on the server (homelab#332, D1). That is an
        ``agtype_in_operator`` over the whole list for each edge row, so the
        read grew with the square of the scope: 12.7 s for the 5,339 nodes of
        the ``homelab`` set on a production-size copy (amd64 emulation),
        against 2.8 s for this form, which reads 1.6 times the induced edges.
        """
        if not node_ids:
            return []
        rows = await self.query(
            f"""
            UNWIND $ids AS wanted
            MATCH (n:{N})-[r:{E}]->(m:{N}) WHERE n.id = wanted
            RETURN n.id AS source, m.id AS target, r.{REL} AS type, properties(r) AS properties
            """,
            {"ids": node_ids},
        )
        wanted = set(node_ids)
        return [row for row in rows if row["target"] in wanted]

    async def get_nodeset_subgraph(
        self, node_type: Type[Any], node_name: List[str], node_name_filter_operator: str = "OR"
    ) -> Tuple[List[Tuple[int, dict]], List[Tuple[int, int, str, dict]]]:
        """The named nodes of one type, their neighbours, and every edge among them.

        ``OR`` takes every neighbour of a named node; ``AND`` takes only the
        neighbours of EVERY named node.
        """
        if not node_name:
            return [], []

        primary_rows = await self.query(
            f"""
            MATCH (n:{N}) WHERE n.type = $type AND n.name IN $names
            RETURN n.id AS id
            """,
            {"type": node_type.__name__, "names": list(node_name)},
        )
        primary_ids = _dedupe(row["id"] for row in primary_rows)
        if not primary_ids:
            return [], []

        pair_rows = await self.query(
            _anchored_edges_query(returns="s.id AS source, t.id AS target"),
            {"anchor_ids": sorted(primary_ids)},
        )
        primary = set(primary_ids)
        anchors_of: Dict[str, Set[str]] = {}
        for row in pair_rows:
            for anchor, neighbour in ((row["source"], row["target"]), (row["target"], row["source"])):
                if anchor in primary:
                    anchors_of.setdefault(neighbour, set()).add(anchor)
        if node_name_filter_operator == "OR":
            neighbours = set(anchors_of)
        else:
            neighbours = {node for node, anchors in anchors_of.items() if anchors == primary}

        all_ids = sorted(primary | neighbours)
        nodes = _node_tuples(await self._nodes_by_id(all_ids))
        edges = [_edge_tuple(row) for row in await self._induced_edge_rows(all_ids)]
        return nodes, edges

    async def get_filtered_graph_data(
        self,
        attribute_filters: List[Dict[str, List[Union[str, int]]]],
        *,
        include_edges: bool = True,
    ) -> Tuple[List[Node], List[EdgeData]]:
        """Nodes whose attributes match every filter, and the edges between them.

        Values are parameters. An attribute with a btree index (``id``,
        ``type``, ``document_id``, ``source_chunk_id``) drives the read through
        that index; cognee filters on ``type`` and ``id``. Without one, the read
        is a scan of the data label.

        🚨 ``include_edges=False`` (cognee 1.6.6) must skip the edge query, not
        filter its result: cognee reads the ``FactConflict`` nodes this way on
        every RAG, chunk and graph search.

        The edge read starts from each matched node (``start_id`` index) and
        filters the target with the same attribute filters, so its cost follows
        the edges of the matched nodes.

        🚨 **A filter with one value is ``n.p = $value``, not an UNWIND**
        (homelab#332, D2). PostgreSQL estimates 1,000 rows for any UNWIND of a
        parameter. On the production graph that estimate turned the
        FactConflict read of every search into a Seq Scan and a hash join over
        all 207,123 nodes (0.59 s). The equality gets the btree index.
        """
        if not attribute_filters:
            return [], []

        params: Dict[str, Any] = {}
        lists: Dict[str, List[Any]] = {}
        clauses: Dict[str, str] = {}
        single: Set[str] = set()
        driver: Optional[str] = None
        for index, (attribute, values) in enumerate(attribute_filters[0].items()):
            param = f"filter_{index}"
            lists[param] = _dedupe(values)
            if len(lists[param]) == 1:
                single.add(param)
                params[param] = lists[param][0]
            else:
                params[param] = lists[param]
            clauses[param] = _quote(attribute)
            if driver is None and attribute in _INDEXED_NODE_PROPERTIES:
                driver = param

        def conditions(alias: str, skip: Optional[str] = None, equality: bool = True) -> List[str]:
            return [
                f"{alias}.{attribute} {'=' if equality and param in single else 'IN'} ${param}"
                for param, attribute in clauses.items()
                if param != skip
            ]

        if driver and driver not in single:
            where = " AND ".join([f"n.{clauses[driver]} = wanted", *conditions("n", driver)])
            node_query = f"""
                UNWIND ${driver} AS wanted
                MATCH (n:{N}) WHERE {where}
                RETURN properties(n) AS properties
            """
        else:
            node_query = f"""
                MATCH (n:{N}) WHERE {" AND ".join(conditions("n"))}
                RETURN properties(n) AS properties
            """
        node_rows = await self.query(node_query, params)
        nodes = _node_tuples(node_rows)
        if not include_edges or not nodes:
            return nodes, []

        # The target filters stay ``IN`` with the list, which no index serves:
        # the read starts from the matched nodes, and an index condition on the
        # target could make the planner start from the target instead (D3).
        edge_rows = await self.query(
            f"""
            UNWIND $ids AS wanted
            MATCH (n:{N})-[r:{E}]->(m:{N})
            WHERE n.id = wanted AND {" AND ".join(conditions("m", equality=False))}
            RETURN n.id AS source, m.id AS target, r.{REL} AS type, properties(r) AS properties
            """,
            {**lists, "ids": sorted({node_id for node_id, _ in nodes})},
        )
        return (nodes, [_edge_tuple(row) for row in edge_rows])

    async def get_graph_metrics(self, include_optional: bool = False) -> Dict[str, Any]:
        """Graph statistics; the all-pairs ones are reported as not computed.

        Components come from two id-only reads plus union-find, which is exact.
        ⚠ ``diameter``, ``avg_shortest_path_length`` and ``avg_clustering`` stay
        ``-1``: all-pairs computations, infeasible at this scale, and ``-1`` is
        this dict's "not computed" convention. Do not read a ``-1`` as a value.
        """
        node_rows = await self.query(f"MATCH (n:{N}) RETURN n.id AS id")
        edge_rows = await self.query(
            f"MATCH (n:{N})-[:{E}]->(m:{N}) RETURN n.id AS source, m.id AS target"
        )
        node_ids = [row["id"] for row in node_rows]
        edge_pairs = [(row["source"], row["target"]) for row in edge_rows]

        num_nodes = len(node_ids)
        num_edges = len(edge_pairs)
        component_sizes = _connected_components(node_ids, edge_pairs)

        metrics: Dict[str, Any] = {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "mean_degree": (2 * num_edges) / num_nodes if num_nodes else None,
            "edge_density": num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0,
            "num_connected_components": len(component_sizes),
            "sizes_of_connected_components": component_sizes,
            "num_selfloops": -1,
            "diameter": -1,
            "avg_shortest_path_length": -1,
            "avg_clustering": -1,
        }

        if include_optional:
            metrics["num_selfloops"] = sum(
                1 for source, target in edge_pairs if source == target
            )

        return metrics

    async def get_triplets_batch(self, offset: int, limit: int) -> List[Dict[str, Any]]:
        """A page of ``(start_node, relationship_properties, end_node)`` triplets."""
        return await self.query(
            f"""
            MATCH (start_node:{N})-[relationship:{E}]->(end_node:{N})
            RETURN properties(start_node) AS start_node,
                   properties(relationship) AS relationship_properties,
                   properties(end_node) AS end_node
            SKIP $offset LIMIT $limit
            """,
            {"offset": int(offset), "limit": int(limit)},
        )
