"""Labels, index definitions and provenance property names.

Kept apart from the adapter so that the migration loader and the homelab tools
can import the names that the adapter writes.

🚨 **One vertex label and one edge label.** FalkorDB stored the cognee type as a
label and the relationship name as an edge type. In AGE each label is a table,
and a pattern without a label reads every label table. Measured on 2026-11-08
(AGE 1.8.0, arm64): 3,000 edge labels made 12,012 relations, and a one-hop
pattern without a type took 485 ms to plan. With one edge label and the name in
an indexed property, the same read took 0.9 ms. So the cognee type is the
``type`` property, and the relationship name is the ``relationship_name``
property.
"""

from __future__ import annotations

# The vertex label of every cognee node.
NODE_LABEL = "Node"

# The edge label of every cognee relationship. The relationship name from the
# LLM is the RELATIONSHIP_PROPERTY of the edge.
EDGE_LABEL = "Edge"
RELATIONSHIP_PROPERTY = "relationship_name"

# Graph-level metadata (provenance version, delete mode) goes on one vertex of
# its own label. It never gets NODE_LABEL, so is_empty() and get_graph_data()
# never see it.
METADATA_LABEL = "GraphMetadata"
METADATA_NODE_ID = "__cognee_graph_metadata__"

# The four provenance fields, stored as native list properties.
PROVENANCE_COLUMNS = (
    "source_ref_keys",
    "source_dataset_ids",
    "source_run_ids",
    "source_run_refs",
)

# The chunk type. The per-document edge lookup sweeps the edges between two
# chunks: cognee's ``create_chunk_associations`` stamps them with the ref of a
# document while both endpoints can belong to other documents.
CHUNK_TYPE = "DocumentChunk"

# Btree expression indexes on vertex properties, as (property, unique).
#
# * ``id`` serves every id lookup. It is UNIQUE, so a concurrent create of one
#   id fails with a unique violation instead of making a second node.
# * ``type`` serves every query that the FalkorDB adapter wrote with a type label.
# * ``document_id`` and ``source_chunk_id`` serve a document prune: it finds the
#   chunks of a document and the summary of each chunk (homelab
#   ``scripts/cognee_stores.py``).
#
# 🚨 A btree entry holds approximately 2.7 kB at most. A value longer than that
# fails the write. Every property here holds an id or a short type name.
NODE_BTREE_INDEXES = (
    ("id", True),
    ("type", False),
    ("document_id", False),
    ("source_chunk_id", False),
)

# GIN expression indexes on vertex list properties. They serve the list lookups
# ``n.source_ref_keys @> [$key]`` and ``n.belongs_to_set ?| $tags``.
NODE_GIN_INDEXES = (
    "source_ref_keys",
    "source_dataset_ids",
    "source_run_ids",
    "belongs_to_set",
)

# GIN expression indexes on edge list properties.
EDGE_GIN_INDEXES = (
    "source_ref_keys",
    "source_dataset_ids",
    "source_run_ids",
)

# Connection defaults, used only when cognee's config leaves the field empty.
# 📌 There is no default for a WRONG value: cognee's unset port sentinel
# (``GraphConfig.graph_database_port`` = 123) is passed through, so a missing
# GRAPH_DATABASE_PORT is a refused connection on port 123 and not a silent
# connection to another server.
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 5432
DEFAULT_GRAPH_NAME = "cognee_graph"

# The connection pool. cognee runs the writes of one cognify concurrently, and
# the shared PostgreSQL app also serves paperclip, so the pool stays small.
POOL_MIN_SIZE = 1
POOL_MAX_SIZE = 8

# A write that fails on a unique violation (two workers create one node or one
# edge) or on a deadlock or serialization failure is run again this many times.
WRITE_ATTEMPTS = 5
