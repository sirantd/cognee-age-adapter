"""Every label table has the btree index on ``id`` that AGE's executor looks for.

🚨 A Cypher SET, REMOVE or DELETE finds its row through a btree index on
column 1 (``id``), else it scans the whole label table (AGE 1.8.0,
``cypher_set.c`` lines 717-799, ``find_usable_btree_index_for_attr``). AGE makes
that index for a vertex label only. Measured on 2026-11-09 with 150k edges: an
edge SET took 67 ms without the index and 0.4 ms with it.

The plan gate (``test_plans.py``) cannot see this scan. It is inside AGE's
"Custom Scan (Cypher Set)" node, and the plan shows no scan of the label table.
So this test reads the catalog.
"""

from __future__ import annotations

import pytest

from cognee_age_adapter.constants import EDGE_LABEL, METADATA_LABEL, NODE_LABEL

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

# The labels of the adapter and the two base tables of AGE.
EXPECTED_TABLES = {NODE_LABEL, EDGE_LABEL, METADATA_LABEL, "_ag_label_vertex", "_ag_label_edge"}

# The conditions of find_usable_btree_index_for_attr: a valid btree index with
# no predicate. The executor needs only ``id`` as the first column; the test
# asks for ``id`` alone.
TABLES_WITH_ID_INDEX = """
SELECT t.tablename,
       coalesce(bool_or(x.indisvalid
                        AND am.amname = 'btree'
                        AND x.indkey::text = '1'
                        AND x.indpred IS NULL), false)
FROM pg_tables t
JOIN pg_namespace n ON n.nspname = t.schemaname
JOIN pg_class c ON c.relnamespace = n.oid AND c.relname = t.tablename
LEFT JOIN pg_index x ON x.indrelid = c.oid
LEFT JOIN pg_class i ON i.oid = x.indexrelid
LEFT JOIN pg_am am ON am.oid = i.relam
WHERE t.schemaname = %s
GROUP BY t.tablename
"""


async def test_every_label_table_has_a_btree_index_on_id(adapter):
    rows = await adapter._sql(TABLES_WITH_ID_INDEX, (adapter._graph_name,))
    tables = {name: indexed for name, indexed in rows}

    # An empty result passes the check below, so the table set comes first.
    assert EXPECTED_TABLES <= tables.keys()
    missing = sorted(name for name, indexed in tables.items() if not indexed)
    assert missing == [], f"no btree index on id: {missing}"
