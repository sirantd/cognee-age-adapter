"""The coercion layer, server-free: what leaves Python on the write and read paths.

The rules are the FalkorDB adapter's rules, kept on purpose (homelab#332, story
28); ``coercion.py`` has the table, and the AGE behaviour behind each rule is
asserted in ``test_coercion_live.py``.

The adapter here runs over a fake pool that records each statement instead of
sending it. The recorded parameter map is the exact JSON that psycopg would
bind as ``$1``.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from cognee.infrastructure.engine import DataPoint

from cognee_age_adapter import AGEAdapter
from cognee_age_adapter.adapter import _quote
from cognee_age_adapter.coercion import NUL, coerce_properties, scrub_nul


class _Ent(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"]}


class _RecordingCursor:
    description = None

    def __init__(self, sent):
        self.sent = sent

    async def execute(self, sql, args=None):
        if "pg_advisory_xact_lock" in sql:
            return  # the write lock (D4) carries keys, not a Cypher parameter map
        self.sent.append((sql, json.loads(args[0]) if args else None))

    async def fetchall(self):
        return []


class _RecordingConnection:
    def __init__(self, sent):
        self.sent = sent

    @asynccontextmanager
    async def transaction(self):
        yield

    @asynccontextmanager
    async def cursor(self):
        yield _RecordingCursor(self.sent)


class _RecordingPool:
    def __init__(self):
        self.sent: list[tuple[str, dict]] = []

    @asynccontextmanager
    async def connection(self):
        yield _RecordingConnection(self.sent)


@pytest.fixture
def adapter():
    """An adapter over a recording pool: real methods, no I/O."""
    instance = AGEAdapter(host="unused")
    instance._pool = _RecordingPool()
    instance._pool_open = True
    return instance


def _params(adapter, index=-1) -> dict:
    return adapter._pool.sent[index][1]


# ----------------------------------------------------------------------
# Maps, and arrays that hold one: JSON-encoded, as on FalkorDB
# ----------------------------------------------------------------------


def test_map_is_json_encoded_under_its_own_key(adapter):
    """AGE stores a map natively, but the rule stays: a migrated node and a
    re-cognified one must have the same form. Encoded under the SAME key, so
    ``_indexed_fields_from_properties`` still finds ``metadata``."""
    out = adapter.serialize_properties({"metadata": {"index_fields": ["name"]}})
    assert json.loads(out["metadata"]) == {"index_fields": ["name"]}


def test_array_containing_a_map_is_json_encoded(adapter):
    out = adapter.serialize_properties({"items": [{"a": 1}]})
    assert json.loads(out["items"]) == [{"a": 1}]


def test_a_map_nested_deep_inside_an_array_is_still_json_encoded(adapter):
    out = adapter.serialize_properties({"items": [1, [2, {"a": 1}]]})
    assert json.loads(out["items"]) == [1, [2, {"a": 1}]]


# ----------------------------------------------------------------------
# Native values: encode nothing the store can hold
# ----------------------------------------------------------------------


def test_array_of_primitives_is_passed_through(adapter):
    assert adapter.serialize_properties({"belongs_to_set": ["a", "b"]}) == {
        "belongs_to_set": ["a", "b"]
    }


@pytest.mark.parametrize(
    "value",
    [[], [0.1, 0.2], [True, False], [[1, 2]], [[["a"]]], [1, ["a"]]],
    ids=["empty", "floats", "bools", "nested", "deep-nested", "heterogeneous"],
)
def test_arrays_without_maps_are_never_encoded(adapter, value):
    assert adapter.serialize_properties({"v": value}) == {"v": list(value)}


def test_uuid_is_stringified(adapter):
    node_id = uuid4()
    assert adapter.serialize_properties({"id": node_id}) == {"id": str(node_id)}


def test_uuid_inside_an_array_is_stringified_too(adapter):
    first, second = uuid4(), uuid4()
    out = adapter.serialize_properties({"ids": [first, second]})
    assert out["ids"] == [str(first), str(second)]


def test_an_unstorable_scalar_falls_back_to_its_string_form(adapter):
    moment = datetime(2026, 8, 9, 12, 0, tzinfo=timezone.utc)
    assert adapter.serialize_properties({"at": moment}) == {"at": str(moment)}


def test_bytes_are_decoded_rather_than_repr_ed(adapter):
    """``str(b"hello")`` would store the repr ``b'hello'``."""
    assert adapter.serialize_properties({"v": b"hello"}) == {"v": "hello"}


def test_nul_inside_bytes_is_gone_once_they_are_decoded(adapter):
    """The transport scrub only sees ``str``, so the decode has to happen first."""
    assert adapter.serialize_properties({"v": b"he\x00llo"}) == {"v": "he\x00llo"}
    assert scrub_nul(adapter.serialize_properties({"v": b"he\x00llo"})) == {"v": "hello"}


def test_weights_are_flattened_not_encoded(adapter):
    """cognee's visualization preprocessor reads ``weight_<name>`` scalars."""
    assert adapter.serialize_properties({"weights": {"trust": 0.5}}) == {"weight_trust": 0.5}


# ----------------------------------------------------------------------
# Nulls: asserted at the layer, never by a round trip
# ----------------------------------------------------------------------


def test_a_null_property_is_dropped_before_the_write():
    """🚨 Asserted HERE, not by writing and reading back: on a fresh node "never
    sent" and "discarded" both read back as absent. On an existing node AGE
    treats a null as a **delete** (``test_coercion_live.py``), so the key must
    not leave Python at all."""
    assert coerce_properties({"a": 1, "b": None}) == {"a": 1}


def test_the_dropped_key_is_absent_rather_than_falsy():
    assert "b" not in coerce_properties({"b": None})


def test_a_falsy_value_that_is_not_null_survives():
    assert coerce_properties({"zero": 0, "empty": "", "none_left": [], "off": False}) == {
        "zero": 0,
        "empty": "",
        "none_left": [],
        "off": False,
    }


def test_nulls_inside_an_array_are_dropped(adapter):
    """Dropped, not JSON-encoded around: ``belongs_to_set`` must stay a list for
    the ``?|`` tag lookup."""
    assert adapter.serialize_properties({"belongs_to_set": ["Dev", None]}) == {
        "belongs_to_set": ["Dev"]
    }


def test_nulls_are_dropped_at_every_depth_of_an_array(adapter):
    assert adapter.serialize_properties({"v": [1, [2, None]]}) == {"v": [1, [2]]}


def test_a_null_inside_a_json_encoded_array_is_preserved(adapter):
    out = adapter.serialize_properties({"items": [{"a": 1}, None]})
    assert json.loads(out["items"]) == [{"a": 1}, None]


# ----------------------------------------------------------------------
# NUL: the one character PostgreSQL text cannot hold
# ----------------------------------------------------------------------


def test_nul_is_stripped_from_a_string():
    assert scrub_nul("a\x00b") == "ab"


@pytest.mark.parametrize("code", [*range(0x01, 0x20), 0x7F])
def test_every_other_control_character_survives(code):
    text = f"a{chr(code)}b"
    assert scrub_nul(text) == text


def test_nul_is_stripped_from_nested_containers_and_from_keys():
    assert scrub_nul({"k\x00ey": ["a\x00b", {"deep": "c\x00d"}]}) == {
        "key": ["ab", {"deep": "cd"}]
    }


def test_scrubbing_returns_the_ORIGINAL_object_when_there_is_nothing_to_scrub():
    """Every query's params go through this, also 1,000-node write batches."""
    params = {"nodes": [{"properties": {"name": "Germany"}}]}
    assert scrub_nul(params) is params


def test_quote_drops_nul_from_an_identifier():
    assert _quote(f"is{NUL}related") == "`isrelated`"


def test_quote_doubles_a_backtick_so_the_name_cannot_end_the_identifier():
    """AGE reads two backticks inside a quoted name as one backtick (measured)."""
    assert _quote("evil`) MATCH (n) DETACH DELETE n //") == (
        "`evil``) MATCH (n) DETACH DELETE n //`"
    )


# ----------------------------------------------------------------------
# What actually leaves Python
# ----------------------------------------------------------------------


async def test_a_datapoints_unset_optionals_never_reach_the_write(adapter):
    """🚨 A bare DataPoint dumps 7 null-valued keys (``source_pipeline`` and
    friends). Passed through, every re-cognify would DELETE them on a node that
    another pipeline had filled in."""
    await adapter.add_nodes([_Ent(id=uuid4(), name="Germany")])

    properties = _params(adapter)["nodes"][0]["properties"]
    assert None not in properties.values()
    assert "source_pipeline" not in properties
    assert properties["name"] == "Germany"
    assert properties["type"] == "_Ent"


async def test_no_nul_reaches_the_store_on_the_write_path(adapter):
    await adapter.add_nodes([_Ent(id=uuid4(), name="Ger\x00many")])
    assert _params(adapter)["nodes"][0]["properties"]["name"] == "Germany"


async def test_lookup_keys_are_scrubbed_the_same_way_the_write_was(adapter):
    """Write and read must scrub identically, or the lookup silently misses."""
    await adapter.get_node("ab\x00cd")
    assert _params(adapter)["node_id"] == "abcd"


async def test_a_relationship_name_is_scrubbed_like_every_other_value(adapter):
    await adapter.add_edge("a", "b", "is\x00related_to")
    edge = _params(adapter)["edges"][0]
    assert edge["rel"] == "isrelated_to"
    assert edge["properties"]["relationship_name"] == "isrelated_to"


async def test_the_edge_property_cannot_change_the_edge_identity(adapter):
    """The relationship name of the edge identity is the third tuple item. A
    different ``relationship_name`` in the property dict must not overwrite it."""
    await adapter.add_edge("a", "b", "knows", {"relationship_name": "other"})
    edge = _params(adapter)["edges"][0]
    assert edge["rel"] == edge["properties"]["relationship_name"] == "knows"
