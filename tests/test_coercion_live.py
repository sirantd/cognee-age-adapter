"""The coercion layer, live half: the AGE behaviours the rules depend on.

Two kinds of test, and the split is the point:

* **What AGE does**, asserted against raw Cypher. These are the measurements
  that ``coercion.py`` cites. If an AGE upgrade changes one, this file fails,
  and the rule gets a new look instead of becoming folklore.
* **What the adapter does about it**: the consequence that a cognify sees.

📌 AGE holds more than FalkorDB could (maps, nulls in arrays). Those rules stay
for the same form as on FalkorDB, not because AGE needs them; the tests below
say which rules are load-bearing on AGE.
"""

from __future__ import annotations

import json
from typing import Optional
from uuid import uuid4

import psycopg
import pytest

from cognee.infrastructure.engine import DataPoint

from cognee_age_adapter.constants import EDGE_LABEL, NODE_LABEL

import conftest

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


class _Ent(DataPoint):
    name: str
    summary: Optional[str] = None
    metadata: dict = {"index_fields": ["name"]}


def _raw(adapter, cypher: str, params: Optional[dict] = None) -> list:
    """Straight to PostgreSQL, past the adapter's own scrubbing."""
    with conftest.connect() as conn:
        rows = conn.execute(
            f"SELECT * FROM ag_catalog.cypher('{adapter._graph_name}', $q${cypher}$q$, %s) "
            "AS (v ag_catalog.agtype)",
            (json.dumps(params or {}),),
        ).fetchall()
    return [row[0] for row in rows]


# ----------------------------------------------------------------------
# What AGE does
# ----------------------------------------------------------------------


async def test_a_null_in_a_merged_map_DELETES_the_stored_property(adapter):
    """🚨 The behaviour that makes null-stripping load-bearing on AGE."""
    _raw(adapter, f"CREATE (:{NODE_LABEL} {{id: 'n1', keep: 'value'}})")

    _raw(adapter, f"MATCH (n:{NODE_LABEL}) WHERE n.id = 'n1' WITH n, $m AS m SET n += m", {"m": {"keep": None}})

    stored = json.loads(_raw(adapter, f"MATCH (n:{NODE_LABEL}) WHERE n.id = 'n1' RETURN properties(n)")[0])
    assert "keep" not in stored


async def test_nul_in_a_parameter_fails_the_whole_statement(adapter):
    """🚨 The behaviour that makes the NUL scrub load-bearing: one NUL in a
    1,000-node batch fails all 1,000."""
    with pytest.raises(psycopg.errors.UntranslatableCharacter):
        _raw(adapter, "RETURN $v", {"v": "a\x00b"})


async def test_every_other_control_character_round_trips(adapter):
    text = "".join(chr(code) for code in (*range(0x01, 0x20), 0x7F))
    assert json.loads(_raw(adapter, "RETURN $v", {"v": text})[0]) == text


async def test_maps_and_nulls_in_arrays_are_stored_natively(adapter):
    """📌 Unlike FalkorDB. The JSON-encode and drop rules stay only for the same
    form as the FalkorDB graph that the migration copies."""
    _raw(
        adapter,
        f"CREATE (:{NODE_LABEL} {{id: 'n2'}})",
    )
    _raw(
        adapter,
        f"MATCH (n:{NODE_LABEL}) WHERE n.id = 'n2' WITH n, $m AS m SET n += m",
        {"m": {"map": {"k": [1, None]}, "list": [1, None]}},
    )
    stored = json.loads(_raw(adapter, f"MATCH (n:{NODE_LABEL}) WHERE n.id = 'n2' RETURN properties(n)")[0])
    assert stored["map"] == {"k": [1, None]}
    assert stored["list"] == [1, None]


# ----------------------------------------------------------------------
# What the adapter does about it
# ----------------------------------------------------------------------


async def test_rewriting_a_node_with_unset_optionals_keeps_the_stored_values(adapter):
    """A second cognify must not strip what the first extracted."""
    node_id = uuid4()
    await adapter.add_nodes([_Ent(id=node_id, name="Germany", summary="a federal republic")])

    await adapter.add_nodes([_Ent(id=node_id, name="Germany")])  # summary unset -> None

    assert (await adapter.get_node(str(node_id)))["summary"] == "a federal republic"


async def test_a_property_holding_nul_is_written_and_reads_back_scrubbed(adapter):
    node_id = uuid4()
    await adapter.add_nodes([_Ent(id=node_id, name="Ger\x00many")])

    assert (await adapter.get_node(str(node_id)))["name"] == "Germany"


async def test_an_array_property_holding_a_null_is_written_without_it(adapter):
    """Reachable from the ``(id, properties)`` tuple and bare-dict forms, which
    no model validates."""
    node_id = str(uuid4())
    await adapter.add_nodes([(node_id, {"type": "Entity", "aliases": ["Deutschland", None]})])

    assert (await adapter.get_node(node_id))["aliases"] == ["Deutschland"]


async def test_an_edge_property_holding_a_null_still_writes(adapter):
    source, target = uuid4(), uuid4()
    await adapter.add_nodes([_Ent(id=source, name="A"), _Ent(id=target, name="B")])

    await adapter.add_edge(
        str(source), str(target), "relates_to", {"evidence": ["chunk-1", None], "score": None}
    )

    _, edges = await adapter.get_graph_data()
    _, _, _, properties = edges[0]
    assert properties["evidence"] == ["chunk-1"]
    assert "score" not in properties


async def test_a_map_valued_property_survives_a_round_trip_as_json(adapter):
    """``metadata`` is a map on every DataPoint, and ``get_node_delete_data``
    parses it back."""
    node_id = uuid4()
    await adapter.add_nodes([_Ent(id=node_id, name="Germany")])

    delete_data = await adapter.get_node_delete_data([str(node_id)])
    assert delete_data[str(node_id)].indexed_fields == ["name"]


async def test_a_relationship_name_holding_nul_does_not_break_the_write(adapter):
    source, target = uuid4(), uuid4()
    await adapter.add_nodes([_Ent(id=source, name="A"), _Ent(id=target, name="B")])

    await adapter.add_edge(str(source), str(target), "is\x00related_to")

    rows = await adapter.query(f"MATCH (:{NODE_LABEL})-[r:{EDGE_LABEL}]->(:{NODE_LABEL}) RETURN r.relationship_name AS rel")
    assert rows == [{"rel": "isrelated_to"}]
