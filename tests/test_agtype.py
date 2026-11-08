"""How an agtype result value becomes plain Python — server-free.

psycopg returns agtype as its text form. The samples here are the exact text
that AGE 1.8.0 returned on 2026-11-08 (probe output), so the parser is checked
against the real format and not against an imagined one.

A vertex or an edge becomes its property map, as neo4j's ``Result.data()`` and
the FalkorDB adapter return them. A path becomes the list of its elements.
"""

from __future__ import annotations

import math

import pytest

from cognee_age_adapter.agtype import parse

VERTEX = (
    '{"id": 844424930131969, "label": "Node", "properties": '
    '{"f": 1.5, "i": 2, "l": [1, "x"], "m": "it\'s", "id": "a"}}::vertex'
)
EDGE = (
    '{"id": 1125899906842625, "label": "EDGE", "end_id": 844424930131970, '
    '"start_id": 844424930131969, "properties": {"relationship_name": "knows"}}::edge'
)


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ('"a"', "a"),
        ("2", 2),
        ("1.5", 1.5),
        ("true", True),
        ("null", None),
        ('[1, "x"]', [1, "x"]),
        ('{"a": [1, {"b": null}]}', {"a": [1, {"b": None}]}),
        ("10::numeric", 10),
        ("1.50::numeric", 1.5),
        ('"a\\u0001b\x7f"', "a\x01b\x7f"),
        ("4611686018427387904", 2**62),
    ],
)
def test_scalars_and_containers(text, value):
    assert parse(text) == value


def test_sql_null_is_none():
    assert parse(None) is None


def test_a_vertex_becomes_its_properties():
    assert parse(VERTEX) == {"f": 1.5, "i": 2, "l": [1, "x"], "m": "it's", "id": "a"}


def test_an_edge_becomes_its_properties():
    assert parse(EDGE) == {"relationship_name": "knows"}


def test_a_path_becomes_the_list_of_its_elements():
    path = f"[{VERTEX}, {EDGE}, {VERTEX}]::path"
    assert parse(path) == [parse(VERTEX), {"relationship_name": "knows"}, parse(VERTEX)]


def test_a_vertex_inside_a_map_is_unwrapped():
    assert parse(f'{{"n": {VERTEX}}}') == {"n": parse(VERTEX)}


def test_a_suffix_inside_a_string_is_text():
    """``::vertex`` in a property value is data, not a type annotation."""
    assert parse('{"s": "a}::vertex", "t": "1::numeric"}') == {"s": "a}::vertex", "t": "1::numeric"}


def test_a_map_with_vertex_keys_stays_a_map():
    """Only the ``::vertex`` annotation makes a vertex, never the key names."""
    value = '{"id": 1, "label": "x", "properties": {"a": 1}}'
    assert parse(value) == {"id": 1, "label": "x", "properties": {"a": 1}}


@pytest.mark.parametrize(("text", "check"), [("NaN", math.isnan), ("Infinity", math.isinf)])
def test_non_finite_floats(text, check):
    assert check(parse(text))
