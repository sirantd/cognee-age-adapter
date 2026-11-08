"""How a Cypher query becomes one SQL statement for AGE — server-free.

AGE runs Cypher through ``SELECT * FROM cypher(graph, query, params) AS (cols)``.
The column list must have one column for each item of the last ``RETURN``, so
the adapter reads the items from the query text. cognee's CYPHER and
NATURAL_LANGUAGE retrievers send LLM-written Cypher through ``query()``, so the
reader must handle the general form, not only the adapter's own queries.

The live half (that AGE accepts what this composes) is every integration test.
"""

from __future__ import annotations

import pytest

from cognee_age_adapter.cypher import compose, return_columns


@pytest.mark.parametrize(
    ("query", "columns"),
    [
        ("MATCH (n) RETURN n", ["n"]),
        ("MATCH (n) RETURN n.id AS id, n.name", ["id", "n.name"]),
        ("MATCH (n) RETURN DISTINCT n.id AS id", ["id"]),
        ("MATCH (n) return n.id as id", ["id"]),
        ("MATCH (n) RETURN n.id AS id ORDER BY id SKIP $o LIMIT $l", ["id"]),
        ("MATCH (n) WITH n RETURN collect(n.id) AS ids, count(*)", ["ids", "count(*)"]),
        ("MATCH (n) RETURN {a: 1, b: [1, 2]} AS m, [x IN [1, 2] | x] AS l", ["m", "l"]),
        ("MATCH (n) RETURN n.id AS `the id`", ["the id"]),
        ("MATCH (n) RETURN n.id AS `a``b`", ["a`b"]),
        ("MATCH (n) RETURN 'RETURN a, b' AS s", ["s"]),
        ("MATCH (n) RETURN \"x, y\" AS s, n.returned AS r", ["s", "r"]),
        ("MATCH (n) RETURN n.id AS id UNION MATCH (m) RETURN m.id AS id", ["id"]),
        ("MATCH (n) RETURN n.id AS id;", ["id"]),
        ("MATCH (n)\nUNWIND keys(n) AS prop\nRETURN DISTINCT labels(n) AS NodeLabels, "
         "collect(DISTINCT prop) AS Properties;\n", ["NodeLabels", "Properties"]),
        ("MATCH (n) SET n.x = 1", []),
        ("MATCH (n) WHERE n.desc = 'no RETURN here' DETACH DELETE n", []),
    ],
)
def test_return_columns_reads_the_last_return(query, columns):
    assert return_columns(query) == columns


def test_return_star_is_refused():
    """AGE needs the column list before it runs the query; ``*`` names none."""
    with pytest.raises(ValueError, match=r"RETURN \*"):
        return_columns("MATCH (n) RETURN *")


def test_compose_wraps_the_query_with_one_column_per_return_item():
    sql = compose("cognee_graph", "MATCH (n:Node) RETURN n.id AS id, n.type AS type")

    assert sql == (
        "SELECT * FROM ag_catalog.cypher('cognee_graph', "
        "$cypher$MATCH (n:Node) RETURN n.id AS id, n.type AS type$cypher$, %s) "
        "AS (c0 ag_catalog.agtype, c1 ag_catalog.agtype)"
    )


def test_compose_gives_a_query_without_return_one_column():
    """AGE refuses an empty column list; a write returns no rows in one column."""
    assert compose("g_x", "MATCH (n) SET n.x = 1").endswith("AS (c0 ag_catalog.agtype)")


def test_compose_escapes_percent_for_the_driver():
    """psycopg reads ``%`` as a placeholder. Measured: ``'100%'`` in a query
    failed with ``only '%s', '%b', '%t' are allowed as placeholders``."""
    assert "'100%%'" in compose("g_x", "RETURN '100%' AS a")


def test_compose_strips_a_trailing_semicolon():
    """Measured: AGE fails ``RETURN n.id;`` with ``syntax error at or near ";"``.
    cognee's own schema queries end with one."""
    assert "RETURN 1 AS a$cypher$" in compose("g_x", "RETURN 1 AS a ;\n")


def test_compose_refuses_the_dollar_tag_in_the_query():
    with pytest.raises(ValueError, match="cypher"):
        compose("g_x", "RETURN '$cypher$' AS a")


@pytest.mark.parametrize("name", ["ab", "1abc", "a-b-c", "x'); DROP TABLE t; --", ""])
def test_compose_refuses_an_invalid_graph_name(name):
    """The graph name goes into the SQL text, so it must be a plain identifier.
    AGE itself refuses names shorter than 3 characters."""
    with pytest.raises(ValueError, match="graph name"):
        compose(name, "RETURN 1 AS a")
