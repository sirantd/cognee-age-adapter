# cognee-age-adapter

A provenance-complete [Apache AGE](https://age.apache.org/) graph adapter for
[cognee](https://github.com/topoteretes/cognee).

It replaces [`cognee-falkordb-adapter`](https://github.com/sirantd/cognee-falkordb-adapter)
in a homelab deployment ([homelab#332](https://github.com/sirantd/homelab/issues/332)).
The graph moves into AGE, a PostgreSQL extension, in a shared PostgreSQL 18
server. The graph gets WAL, MVCC, `statement_timeout`, `pg_dump` and
disk-resident storage.

## Status

**Stage A of homelab#332: not deployed.** The adapter is a port of
cognee-falkordb-adapter 0.5.4, with the same public surface, provenance
semantics and value rules. The CI gate runs against the production store:

* cognee's own provenance contract suite must pass 20/20 (cognee `1.6.8`).
* Every query that the adapter emits must read its label tables through an
  index (see [The plan gate](#the-plan-gate)).

On 2026-11-08 the full suite passed locally against `apache/age` and against
the production image under amd64 emulation.

The FalkorDB adapter stays the rollback path until homelab#332 Stage F.

## Use

Install the adapter into the cognee image, import the registration once, and
set the graph provider:

```python
import cognee_age_adapter.register  # binds the "age" graph provider
```

| variable | value |
|---|---|
| `GRAPH_DATABASE_PROVIDER` | `age` |
| `GRAPH_DATABASE_URL` | the PostgreSQL host, or a `postgresql://` URL |
| `GRAPH_DATABASE_PORT` | `5432` |
| `GRAPH_DATABASE_USERNAME` / `GRAPH_DATABASE_PASSWORD` | the graph role |
| `GRAPH_DATABASE_NAME` | the AGE graph name, for example `cognee_graph` |
| `VECTOR_DB_PROVIDER` | `pgvector`; the registration refuses any other vector store |

The PostgreSQL database is the database of a `postgresql://` URL. With a bare
host, the database has the name of the graph. So production uses one name,
`cognee_graph`, for the database and for the graph.

⚠ cognee constructs the adapter with its own keyword names
(`graph_database_url`, `graph_database_port`, `database_name`, ...).
`tests/test_factory.py` reads that call from cognee's source, so a renamed
keyword fails CI and not the first cognify.

### What the server must have

1. AGE 1.8.0 available to the cluster. In production the files come from a
   read-only mount (`ci/pins.env`, `ci/fetch-age.sh`) and two settings
   (`ci/init-cluster.sql`).
2. A graph database with the extension, and a graph role that is **not** a
   superuser. The role's own settings hold the AGE preload, the search path, a
   `statement_timeout` and an `idle_in_transaction_session_timeout`
   (`ci/init-graph-db.sql`). The search path is
   load-bearing: AGE's own label DDL fails without `ag_catalog` in it.

🚨 The adapter never sends `LOAD 'age'`. For a non-superuser it fails with
`access to library "age" is not allowed`. AGE comes from the role's
`session_preload_libraries`, for the graph database only.

`initialize()` creates the graph, its three labels and every index, if they
are absent. cognee calls it when it builds the engine. The role needs no
privilege on AGE's catalog tables.

## Store layout

| | AGE | FalkorDB adapter |
|---|---|---|
| cognee node | one vertex label `Node`; the cognee type in the `type` property | the type as a label |
| relationship | one edge label `Edge`; the name in the `relationship_name` property | the name as an edge type |
| graph metadata | one vertex of the label `GraphMetadata` | one node of its own label |

🚨 One label each, because AGE makes one table for each label. Measured on
2026-11-08: 3,000 edge labels made 12,012 relations, and one hop without a
type took 485 ms to plan. `cognee_graph` has approximately 23,500 relationship
names. As property values they make no tables, and one read takes any number of
names (the FalkorDB 255-type limit is gone).

### Indexes

Btree expression indexes on the vertex properties `id` (unique), `type`,
`document_id` and `source_chunk_id`. GIN expression indexes on the provenance
lists and `belongs_to_set`. On the edges: a unique index on the identity
`(start_id, end_id, relationship_name)`, a btree index on `relationship_name`,
a btree index on the `id` column, and GIN indexes on the provenance lists.

AGE makes the `id` index of a vertex label itself. For an edge label, AGE makes
only the `start_id` and `end_id` indexes, so the adapter makes the `id` index.

🚨 A Cypher `SET` finds its row through a btree index on `id`. Without that
index, each edge `SET` scans the whole `Edge` table. The plan gate cannot see
this scan, so `tests/test_indexes.py` reads the catalog.

🚨 A btree entry holds approximately 2.7 kB. A relationship name longer than
that fails its write. Measure the longest name before a migration.

🚨 The edge identity index is unique. If the FalkorDB graph holds two edges
with one source, one target and one relationship name, the second fails the
load. The migration must count such duplicates first (homelab#332, Stage D).

## Query rules

Each rule is a measurement on AGE 1.8.0, and each has a test.

* A lookup by a list of values is `UNWIND $values AS v ... WHERE n.p = v`.
  `WHERE n.p IN $values` compiles to `agtype_in_operator`, which no index
  serves. `IN` stays only as a filter after an index-backed step.
* An `IN` against a long list reads the whole list for each row. The read of
  the edges among a scope of 5,339 nodes took 12.7 s with `m.id IN $ids`, and
  2.8 s with the target filter in Python (homelab#332, D1; a production-size
  copy under amd64 emulation).
* A filter with one value is `WHERE n.p = $value`, not an `UNWIND`.
  PostgreSQL estimates 1,000 rows for every `UNWIND` of a parameter. At
  production size that estimate read all 207,123 nodes for the FactConflict
  filter of every search (homelab#332, D2).
* A read that starts from given nodes has no other condition that an index
  serves for many rows. The planner estimated 818 rows for the dataset GIN
  condition `r.source_dataset_ids @> [$id]` (651,981 actual), started from
  it, and took 138 s for a document of 22 nodes (homelab#332, D3). Such a
  condition goes after the anchor, in a form that no index serves (`$id IN`).
* A match by edge identity is two statements. A SQL statement finds the
  graphid of each edge, with each end as a scalar subquery on the unique `id`
  index, so both ends are known before the edge scan. The Cypher statement
  then matches `id(r)`, which `edge_id_btree` serves. Every Cypher form of the
  identity match plans as a join: the planner estimated 9 edges for each
  source and read all 1,271 edges of the largest source through the
  `start_id` index (homelab#332, D5).
* ⚠ The index of that edge scan is a cost decision: AGE's `start_id` and
  `end_id` indexes also serve its condition. On the production-size copy the
  identity index costs 25.4, a BitmapAnd of the two AGE indexes 30.7, and
  either one alone approximately 57. The costs tie when each end has
  approximately one edge. Both plan gates assert the identity index.
* A pattern map `{id: $x}` compiles to `properties @> {...}`, which only a GIN
  index over all properties serves. Lookups use `WHERE` instead.
* A vertex `MERGE` matches with such a pattern map. So `add_nodes` creates the
  absent ids, then updates every node through the unique `id` index, in one
  transaction.
* An undirected pattern `(a)-[r]-(b)` plans as an `OR` join over all vertices.
  Each undirected read is two directed reads joined by `UNION`.
* A query that starts with an `OPTIONAL MATCH` that matches nothing returns no
  row, not one row of nulls. Start such a query with `UNWIND`.
* `SET n += <expression>` needs a variable: `WITH n, $map AS m SET n += m`.
* AGE refuses a trailing `;`, and `RETURN *` cannot run: AGE needs the column
  list before the query runs. `query()` reads the columns from the last
  `RETURN`.
* A label is created in a statement of its own. 3,000 labels created in one
  transaction failed with `out of shared memory`.

### Concurrent writers

cognify workers in several processes can write the same node or edge at the
same moment. A NodeSet node carries the key of every document of its set, so
every document of one set writes it.

🚨 AGE does not re-check a row that a concurrent transaction updated, as a
native `UPDATE` does. The `SET` waits for the other commit and then fails with
`XX000 Entity failed to be updated: 3`. On 2026-11-10, with 5 writers on
production, 9 of 80 batches failed after five attempts, and their nodes and
edges were lost (homelab#332, D4).

So every node and edge write takes a transaction-level advisory lock for each
node id or edge identity, in sorted order, in its first statement
(`pg_advisory_xact_lock`). A second writer of the same node waits for the first
commit, and its Cypher statements then read the committed row. An edge locks
its identity, not its end nodes: every edge of a set ends at the one NodeSet
node. The locked writes are `add_nodes`, `add_edges`, the attach and remove of
source refs, and `set_node_truth_state`. An attach or remove reads, changes and
writes the provenance in one locked transaction, so a writer in another process
cannot commit between the read and the write.

🚨 The advisory locks of all sessions share the lock table of the server, for
every database of the app. In a local container of the production image, 14,000
locks fit and 16,000 failed with `out of shared memory`. So one transaction
locks at most 256 nodes or edges (`LOCK_BATCH`), and a larger batch is written
in transactions of that size. Each node stays atomic; a batch is not.

The unique indexes stay. A write that still loses a race (a unique violation, a
deadlock, a serialization failure, or `XX000` from a writer without the locks,
such as `remove_belongs_to_set_tags`) runs again, up to five times.

## Coercion

`coercion.py` keeps the rules of the FalkorDB adapter unchanged. Then the
nodes that a migration copies and the nodes that a re-cognify writes have the
same form. Two rules are load-bearing on AGE:

* A null in `SET n += {...}` deletes the property, so null-valued keys never
  leave Python.
* A NUL in a parameter fails the whole statement, so the adapter strips NUL
  from the parameters of every statement.

AGE stores maps and nulls in arrays natively. The adapter still encodes the
maps and drops the nulls, for the same form as on FalkorDB.
`tests/test_coercion_live.py` asserts each behaviour.

## The plan gate

`tests/test_plans.py` calls every public method on a seeded graph, records each
statement at the one place the adapter sends SQL, and asks PostgreSQL for the
plan of exactly that statement with its parameters. With sequential scans, hash
joins and merge joins disabled, the plan must have no disabled node, and every
scan of a label table must carry an `Index Cond` or a `Recheck Cond`. A census
fails when a public method is missing from the workload. Five methods read the
whole graph by design, and they are listed by name.

### At production size

🚨 The CI gate cannot see a plan that the planner picks only at production
size: D2 and D3 of homelab#332 passed it. `tests/test_plans_production.py` runs
the reads of the live path and of the per-document delete on a copy of the
production graph, with the default planner settings. It plans each statement
with its parameters and again as a generic plan (psycopg prepares a statement
after five runs). It limits the label rows that each statement reads:

| statement | may read |
|---|---|
| with an id list (`$anchor_ids`, `$ids`) | 3 × (the ids + the edges of those nodes) + 1,000 |
| dataset-wide by design (the node pass and the chunk sweep of the per-document lookups) | each label once |
| any other | 3 × the rows it returns + 1,000 |

"No Seq Scan" is not the check: the D3 plan was index-backed and still read the
whole dataset. The test reads only, takes its inputs from the graph, and is
skipped unless `AGE_COPY_GRAPH` names the copy. See [Testing](#on-a-production-size-copy).
Without a copy, a census still runs: each public read is probed or excluded by
name.

The gate found D5 of homelab#332: a read by edge identity read every outgoing
edge of the source (20 edges of the node with the most outgoing edges: 25,551
label rows). The edge reads take 20 edges of that source and 20 edges of the
NodeSet with the most incoming edges. A defect that the gate finds and that is
still open goes into `KNOWN_DEFECTS` as a strict `xfail`.

## Testing

### Local

Start a local AGE server and make the graph role:

```sh
docker run -d --name age-dev -e POSTGRES_PASSWORD=postgres -p 55432:5432 apache/age:release_PG18_1.8.0
docker cp ci/init-graph-db.sql age-dev:/tmp/init.sql
docker exec age-dev psql -U postgres -v ON_ERROR_STOP=1 \
  -v role=cognee_graph -v password=cognee_graph -v database=cognee_graph -f /tmp/init.sql
```

Install and run the tests:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[test]'
.venv/bin/pytest -m "not integration"
AGE_PORT=55432 AGE_REQUIRED=1 .venv/bin/pytest
```

For the production layout on a laptop, use the CI scripts. On arm64 they need
amd64 emulation:

```sh
ci/fetch-age.sh .ci-age
AGE_CONTAINER=age-prod AGE_PORT=55433 DOCKER_PLATFORM=linux/amd64 ci/start-postgres.sh .ci-age
AGE_PORT=55433 AGE_REQUIRED=1 .venv/bin/pytest
```

### On a production-size copy

Load a copy of the production graph into the production layout above (the
homelab `scripts/age-migration/cognee_copy.py load`, see homelab
`docs/age-migration.md`), run `ANALYZE` on `Node` and `Edge`, then:

```sh
AGE_PORT=55433 AGE_REQUIRED=1 AGE_COPY_GRAPH=cognee_graph .venv/bin/pytest tests/test_plans_production.py
```

The rest of the suite makes its own throwaway graphs and does not touch the
copy. 🚨 Never point the `adapter` fixture or the workload of
`tests/test_plans.py` at the copy: the workload calls `delete_graph`.

### CI

* The `surface` job runs the server-free tests.
* The `contract` job runs the integration tests on the production image digest,
  as uid 999 with its `PGDATA` and command. AGE is mounted read-only, and the
  tests connect as a non-superuser graph role. `AGE_REQUIRED=1` turns an absent
  server into a failure, not 20 skips.
* The `age-source` workflow proves that the pinned PGDG package is the ASF
  1.8.0 release (`ci/verify-age-source.sh`). It runs when a pin changes.

### Pins

The pins are in `pyproject.toml` (cognee) and `ci/pins.env` (the PostgreSQL
image and the AGE package). `tests/test_contract_pin.py` fails when the cognee
pin or the 20 contract cases change.

⚠ PGDG moves superseded packages from `apt.postgresql.org` to
`apt-archive.postgresql.org`. If `ci/fetch-age.sh` fails with a 404, point
`AGE_DEB_URL` at the archive; the sha256 pin stays.
