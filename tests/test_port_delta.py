"""The port delta from the FalkorDB adapter, against a live AGE.

Every test here covers a behaviour that the store layout changed: the cognee
type and the relationship name are properties now, not labels; vertex writes
are a create-then-update; undirected reads are two directed reads. The
provenance surface itself is cognee's contract suite (``test_contract.py``).
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from cognee.infrastructure.databases.provenance import EdgeIdentity
from cognee.infrastructure.engine import DataPoint

from cognee_age_adapter.constants import EDGE_LABEL, NODE_LABEL

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


class _Ent(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"]}


class _Other(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"]}


class NodeSet(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"]}


async def _label_tables(adapter) -> set[str]:
    rows = await adapter._sql(
        "SELECT tablename FROM pg_tables WHERE schemaname = %s", (adapter._graph_name,)
    )
    return {row[0] for row in rows}


# ----------------------------------------------------------------------
# The cognee type is a property of the one vertex label
# ----------------------------------------------------------------------


async def test_a_node_has_the_one_label_and_its_type_as_a_property(adapter):
    node = _Ent(id=uuid4(), name="Germany")
    await adapter.add_nodes([node])

    rows = await adapter.query(
        f"MATCH (n:{NODE_LABEL}) WHERE n.id = $id RETURN label(n) AS label, n.type AS type",
        {"id": str(node.id)},
    )
    assert rows == [{"label": NODE_LABEL, "type": "_Ent"}]


async def test_a_mixed_type_batch_types_each_node_correctly(adapter):
    ent, other = _Ent(id=uuid4(), name="A"), _Other(id=uuid4(), name="B")
    await adapter.add_nodes([ent, other])

    rows = await adapter.query(f"MATCH (n:{NODE_LABEL}) RETURN n.id AS id, n.type AS type")
    assert {row["id"]: row["type"] for row in rows} == {str(ent.id): "_Ent", str(other.id): "_Other"}


async def test_retyping_a_node_keeps_one_row_for_its_id(adapter):
    """🚨 The duplicate-id trap. The unique ``id`` index makes it impossible
    rather than merely avoided."""
    node_id = uuid4()
    await adapter.add_nodes([_Ent(id=node_id, name="A")])
    await adapter.add_nodes([_Other(id=node_id, name="A")])

    rows = await adapter.query(f"MATCH (n:{NODE_LABEL}) RETURN n.id AS id, n.type AS type")
    assert rows == [{"id": str(node_id), "type": "_Other"}]


async def test_many_relationship_names_make_no_new_tables(adapter):
    """🚨 The reason for the one-edge-label design: cognee_graph has
    approximately 23,500 relationship names. On FalkorDB each was a type; here
    each is a value, and the graph keeps its five tables."""
    nodes = [_Ent(id=uuid4(), name=str(index)) for index in range(2)]
    await adapter.add_nodes(nodes)
    before = await _label_tables(adapter)

    await adapter.add_edges(
        [(str(nodes[0].id), str(nodes[1].id), f"relation {index}", {}) for index in range(50)]
    )

    assert await _label_tables(adapter) == before
    assert {NODE_LABEL, EDGE_LABEL} <= before


# ----------------------------------------------------------------------
# belongs_to_set is a union across writes
# ----------------------------------------------------------------------


async def test_belongs_to_set_is_a_union_across_writes(adapter):
    node_id = uuid4()
    await adapter.add_nodes([_Ent(id=node_id, name="A", belongs_to_set=["Dev"])])
    await adapter.add_nodes([_Ent(id=node_id, name="A", belongs_to_set=["Prod"])])

    assert (await adapter.get_node(str(node_id)))["belongs_to_set"] == ["Dev", "Prod"]


async def test_belongs_to_set_union_does_not_duplicate_an_existing_tag(adapter):
    node_id = uuid4()
    await adapter.add_nodes([_Ent(id=node_id, name="A", belongs_to_set=["Dev", "Prod"])])
    await adapter.add_nodes([_Ent(id=node_id, name="A", belongs_to_set=["Dev"])])

    assert (await adapter.get_node(str(node_id)))["belongs_to_set"] == ["Dev", "Prod"]


async def test_duplicate_ids_within_one_batch_keep_every_tag(adapter):
    """A repeated id in one batch would also fail the create step: the second
    create of one id is a unique violation. The batch is deduplicated first."""
    node_id = uuid4()
    await adapter.add_nodes(
        [
            _Ent(id=node_id, name="A", belongs_to_set=["Dev"]),
            _Ent(id=node_id, name="A", belongs_to_set=["Prod"]),
        ]
    )

    assert sorted((await adapter.get_node(str(node_id)))["belongs_to_set"]) == ["Dev", "Prod"]


# ----------------------------------------------------------------------
# The relationship name is a property of the one edge label
# ----------------------------------------------------------------------


async def test_edges_of_different_names_are_kept_apart(adapter):
    a, b = _Ent(id=uuid4(), name="A"), _Ent(id=uuid4(), name="B")
    await adapter.add_nodes([a, b])
    await adapter.add_edges(
        [
            (str(a.id), str(b.id), "knows", {"edge_text": "t"}),
            (str(a.id), str(b.id), "trusts", {"edge_text": "t2"}),
        ]
    )

    rows = await adapter.query(f"MATCH ()-[r:{EDGE_LABEL}]->() RETURN r.relationship_name AS rel")
    assert sorted(row["rel"] for row in rows) == ["knows", "trusts"]


async def test_a_relationship_name_that_is_not_an_identifier_round_trips(adapter):
    """Extraction emits names with spaces, punctuation and backticks; as a
    property value each one needs no quoting at all."""
    a, b = _Ent(id=uuid4(), name="A"), _Ent(id=uuid4(), name="B")
    await adapter.add_nodes([a, b])
    for name in ("is related to", "работает в", "has`tick"):
        await adapter.add_edges([(str(a.id), str(b.id), name, {"edge_text": "t"})])

        edge = EdgeIdentity(str(a.id), str(b.id), name)
        assert await adapter.has_edge(str(a.id), str(b.id), name) is True
        assert edge in await adapter.get_edge_delete_data([edge])


async def test_re_adding_an_edge_updates_it_and_keeps_its_first_created_at(adapter):
    a, b = _Ent(id=uuid4(), name="A"), _Ent(id=uuid4(), name="B")
    await adapter.add_nodes([a, b])
    await adapter.add_edges([(str(a.id), str(b.id), "knows", {"edge_text": "first"})])
    first = (await adapter.query(f"MATCH ()-[r:{EDGE_LABEL}]->() RETURN properties(r) AS p"))[0]["p"]

    await adapter.add_edges([(str(a.id), str(b.id), "knows", {"edge_text": "second"})])
    rows = await adapter.query(f"MATCH ()-[r:{EDGE_LABEL}]->() RETURN properties(r) AS p")

    assert len(rows) == 1
    assert rows[0]["p"]["edge_text"] == "second"
    assert rows[0]["p"]["created_at"] == first["created_at"]


async def test_a_repeated_edge_in_one_batch_is_one_edge_with_the_later_properties(adapter):
    a, b = _Ent(id=uuid4(), name="A"), _Ent(id=uuid4(), name="B")
    await adapter.add_nodes([a, b])
    await adapter.add_edges(
        [
            (str(a.id), str(b.id), "knows", {"edge_text": "first", "kept": 1}),
            (str(a.id), str(b.id), "knows", {"edge_text": "second"}),
        ]
    )

    rows = await adapter.query(f"MATCH ()-[r:{EDGE_LABEL}]->() RETURN properties(r) AS p")
    assert len(rows) == 1
    assert (rows[0]["p"]["edge_text"], rows[0]["p"]["kept"]) == ("second", 1)


async def test_an_edge_to_an_absent_node_is_not_written(adapter):
    a = _Ent(id=uuid4(), name="A")
    await adapter.add_nodes([a])
    await adapter.add_edges([(str(a.id), str(uuid4()), "knows", {})])

    assert await adapter.query(f"MATCH ()-[r:{EDGE_LABEL}]->() RETURN r.relationship_name AS r") == []


async def test_has_edges_returns_the_subset_that_exists(adapter):
    a, b = _Ent(id=uuid4(), name="A"), _Ent(id=uuid4(), name="B")
    await adapter.add_nodes([a, b])
    await adapter.add_edges([(str(a.id), str(b.id), "knows", {})])

    present = (str(a.id), str(b.id), "knows", {})
    absent = (str(b.id), str(a.id), "knows", {})
    assert await adapter.has_edges([present, absent]) == [present]


# ----------------------------------------------------------------------
# Metrics and truth state
# ----------------------------------------------------------------------


async def test_metrics_count_components(adapter):
    a, b, island = (_Ent(id=uuid4(), name=name) for name in "ABC")
    await adapter.add_nodes([a, b, island])
    await adapter.add_edges([(str(a.id), str(b.id), "knows", {})])

    metrics = await adapter.get_graph_metrics()
    assert metrics["num_nodes"] == 3
    assert metrics["num_edges"] == 1
    assert metrics["num_connected_components"] == 2
    assert metrics["sizes_of_connected_components"] == [2, 1]


async def test_all_pairs_metrics_are_reported_as_not_computed(adapter):
    metrics = await adapter.get_graph_metrics(include_optional=True)
    assert metrics["diameter"] == -1
    assert metrics["avg_shortest_path_length"] == -1
    assert metrics["avg_clustering"] == -1
    assert metrics["num_selfloops"] == 0  # this one IS computed when asked


async def test_truth_state_round_trips(adapter):
    node = _Ent(id=uuid4(), name="A")
    await adapter.add_nodes([node])
    node_id = str(node.id)

    assert await adapter.set_node_truth_state(
        {node_id: {"truth_alignment": ["supported"], "truth_epoch": 3}}
    ) == {node_id: True}
    assert await adapter.get_node_truth_state([node_id]) == {
        node_id: {"truth_alignment": ["supported"], "truth_epoch": 3}
    }


async def test_truth_state_defaults_for_a_node_that_has_none(adapter):
    node = _Ent(id=uuid4(), name="A")
    await adapter.add_nodes([node])

    assert await adapter.get_node_truth_state([str(node.id)]) == {
        str(node.id): {"truth_alignment": [], "truth_epoch": None}
    }


async def test_truth_state_without_an_epoch_leaves_the_stored_one_alone(adapter):
    node = _Ent(id=uuid4(), name="A")
    await adapter.add_nodes([node])
    node_id = str(node.id)

    await adapter.set_node_truth_state({node_id: {"truth_alignment": ["a"], "truth_epoch": 7}})
    await adapter.set_node_truth_state({node_id: {"truth_alignment": ["b"]}})

    assert await adapter.get_node_truth_state([node_id]) == {
        node_id: {"truth_alignment": ["b"], "truth_epoch": 7}
    }


async def test_truth_state_reports_a_missing_node_as_not_updated(adapter):
    assert await adapter.set_node_truth_state({"ghost": {"truth_alignment": []}}) == {
        "ghost": False
    }


# ----------------------------------------------------------------------
# The generic query path
# ----------------------------------------------------------------------


async def test_query_returns_rows_keyed_by_column_name(adapter):
    node = _Ent(id=uuid4(), name="Germany")
    await adapter.add_nodes([node])

    rows = await adapter.query(f"MATCH (n:{NODE_LABEL}) RETURN n.id AS id, n.name AS name")
    assert rows == [{"id": str(node.id), "name": "Germany"}]


async def test_query_unwraps_a_returned_node_to_its_properties(adapter):
    await adapter.add_nodes([_Ent(id=uuid4(), name="Germany")])

    rows = await adapter.query(f"MATCH (n:{NODE_LABEL}) RETURN n")
    assert rows[0]["n"]["name"] == "Germany"


async def test_query_refuses_return_star(adapter):
    with pytest.raises(ValueError, match=r"RETURN \*"):
        await adapter.query(f"MATCH (n:{NODE_LABEL}) RETURN *")


async def test_query_keeps_a_percent_sign_in_the_query_text(adapter):
    assert await adapter.query("RETURN '100%' AS share") == [{"share": "100%"}]


async def test_a_repeated_lookup_stays_correct_past_the_prepared_statement_threshold(adapter):
    """psycopg prepares a statement after 5 runs, and PostgreSQL may change to a
    generic plan after 5 more. The plan tests see only custom plans, where the
    parameter is a constant; this proves the generic one returns the right rows."""
    nodes = [_Ent(id=uuid4(), name=f"n{index}") for index in range(12)]
    await adapter.add_nodes(nodes)

    for node in nodes:
        assert (await adapter.get_node(str(node.id)))["name"] == node.name
    assert await adapter.get_node(str(uuid4())) is None


# ----------------------------------------------------------------------
# Whole-graph reads and delete
# ----------------------------------------------------------------------


async def test_graph_data_falls_back_to_matched_endpoints(adapter):
    """An edge that a bulk loader wrote can lack ``source_node_id``."""
    a, b = _Ent(id=uuid4(), name="A"), _Ent(id=uuid4(), name="B")
    await adapter.add_nodes([a, b])
    await adapter.query(
        f"""
        MATCH (x:{NODE_LABEL}) WHERE x.id = $a
        MATCH (y:{NODE_LABEL}) WHERE y.id = $b
        CREATE (x)-[:{EDGE_LABEL} {{relationship_name: 'knows'}}]->(y)
        """,
        {"a": str(a.id), "b": str(b.id)},
    )

    nodes, edges = await adapter.get_graph_data()
    assert len(nodes) == 2
    assert edges == [(str(a.id), str(b.id), "knows", {"relationship_name": "knows"})]


async def test_graph_data_never_returns_the_metadata_vertex(adapter):
    await adapter.set_graph_metadata({"graph_delete_mode": "graph_provenance"})

    nodes, _ = await adapter.get_graph_data()
    assert nodes == []
    assert await adapter.is_empty() is True


async def test_delete_graph_empties_the_graph_but_keeps_the_indexes(adapter):
    """🚨 ``drop_graph`` would drop the indexes with the data, and nothing on the
    write path makes them again."""
    await adapter.add_nodes([_Ent(id=uuid4(), name="A")])
    await adapter.delete_graph()

    assert await adapter.is_empty() is True
    indexes = await adapter._sql(
        "SELECT indexname FROM pg_indexes WHERE schemaname = %s", (adapter._graph_name,)
    )
    assert "node_id_btree" in {row[0] for row in indexes}


async def test_filtered_graph_data_parameterizes_its_values(adapter):
    """One apostrophe in an entity name must not be a syntax error."""
    await adapter.add_nodes([_Ent(id=uuid4(), name="O'Brien")])

    nodes, _ = await adapter.get_filtered_graph_data([{"name": ["O'Brien"]}])
    assert [properties["name"] for _, properties in nodes] == ["O'Brien"]


async def test_filtered_graph_data_by_type_and_by_id(adapter):
    """The two filters cognee sends; both go through an index."""
    a, b, other = _Ent(id=uuid4(), name="A"), _Ent(id=uuid4(), name="B"), _Other(id=uuid4(), name="C")
    await adapter.add_nodes([a, b, other])
    await adapter.add_edges(
        [(str(a.id), str(b.id), "knows", {}), (str(a.id), str(other.id), "knows", {})]
    )

    nodes, edges = await adapter.get_filtered_graph_data([{"type": ["_Ent"]}])
    assert sorted(properties["name"] for _, properties in nodes) == ["A", "B"]
    assert [(edge[0], edge[1]) for edge in edges] == [(str(a.id), str(b.id))]

    nodes, _ = await adapter.get_filtered_graph_data([{"id": [str(a.id)], "type": ["_Ent"]}])
    assert [properties["name"] for _, properties in nodes] == ["A"]


async def test_filtered_graph_data_without_edges_runs_only_the_node_query(adapter, monkeypatch):
    """cognee 1.6.6 reads FactConflict nodes with ``include_edges=False`` on every
    search. The edge query must not run at all."""
    a, b = _Ent(id=uuid4(), name="A"), _Ent(id=uuid4(), name="B")
    await adapter.add_nodes([a, b])
    await adapter.add_edges([(str(a.id), str(b.id), "knows", {"edge_text": "t"})])

    queries = []
    real_query = adapter.query

    async def recording_query(query, params=None):
        queries.append(query)
        return await real_query(query, params)

    monkeypatch.setattr(adapter, "query", recording_query)
    nodes, edges = await adapter.get_filtered_graph_data(
        [{"name": ["A", "B"]}], include_edges=False
    )

    assert sorted(properties["name"] for _, properties in nodes) == ["A", "B"]
    assert edges == []
    assert len(queries) == 1

    _, edges = await adapter.get_filtered_graph_data([{"name": ["A", "B"]}])
    assert [edge[2] for edge in edges] == ["knows"]


async def test_remove_belongs_to_set_tags_also_prunes_the_stale_nodeset_edge(adapter):
    """The contract suite checks the property; the edge half is ours to prove."""
    node = _Ent(id=uuid4(), name="Shared", belongs_to_set=["Dev"])
    nodeset = NodeSet(id=uuid4(), name="Dev")
    await adapter.add_nodes([node, nodeset])
    await adapter.add_edge(str(node.id), str(nodeset.id), "belongs_to_set")

    await adapter.remove_belongs_to_set_tags(["Dev"])

    assert (await adapter.get_node(str(node.id)))["belongs_to_set"] == []
    assert await adapter.get_edges(str(node.id)) == []


# ----------------------------------------------------------------------
# The read surface, on a small graph
# ----------------------------------------------------------------------


@pytest.fixture
async def triangle(adapter):
    """a -knows-> b -likes-> c, plus an unconnected d."""
    nodes = [_Ent(id=uuid4(), name=name) for name in ("a", "b", "c", "d")]
    await adapter.add_nodes(nodes)
    a, b, c, _ = (str(node.id) for node in nodes)
    await adapter.add_edges(
        [(a, b, "knows", {"edge_text": "ab"}), (b, c, "likes", {"edge_text": "bc"})]
    )
    return {node.name: str(node.id) for node in nodes}


async def test_get_neighbors_is_undirected(adapter, triangle):
    names = {node["name"] for node in await adapter.get_neighbors(triangle["b"])}
    assert names == {"a", "c"}


async def test_get_edges_returns_both_directions(adapter, triangle):
    edges = await adapter.get_edges(triangle["b"])
    assert sorted(properties["relationship_name"] for _, _, properties in edges) == [
        "knows",
        "likes",
    ]


async def test_get_connections_orients_each_triple(adapter, triangle):
    triples = await adapter.get_connections(triangle["b"])
    assert sorted((s["name"], r["relationship_name"], t["name"]) for s, r, t in triples) == [
        ("a", "knows", "b"),
        ("b", "likes", "c"),
    ]


async def test_get_neighborhood_respects_depth(adapter, triangle):
    one_hop, _ = await adapter.get_neighborhood([triangle["a"]], depth=1)
    two_hop, edges = await adapter.get_neighborhood([triangle["a"]], depth=2)

    assert {properties["name"] for _, properties in one_hop} == {"a", "b"}
    assert {properties["name"] for _, properties in two_hop} == {"a", "b", "c"}
    assert len(edges) == 2


async def test_get_neighborhood_filters_by_edge_type(adapter, triangle):
    nodes, _ = await adapter.get_neighborhood([triangle["a"]], depth=2, edge_types=["knows"])
    assert {properties["name"] for _, properties in nodes} == {"a", "b"}


async def test_get_neighborhood_follows_every_listed_type(adapter, triangle):
    nodes, _ = await adapter.get_neighborhood(
        [triangle["a"]], depth=2, edge_types=["knows", "likes"]
    )
    assert {properties["name"] for _, properties in nodes} == {"a", "b", "c"}


async def test_get_neighborhood_follows_edges_in_both_directions(adapter, triangle):
    nodes, _ = await adapter.get_neighborhood([triangle["c"]], depth=2)
    assert {properties["name"] for _, properties in nodes} == {"a", "b", "c"}


async def test_get_neighborhood_returns_every_edge_between_its_nodes(adapter, triangle):
    """``edge_types`` selects the neighbours, not the returned edges."""
    await adapter.add_edges([(triangle["b"], triangle["a"], "trusts", {"edge_text": "ba"})])

    _, edges = await adapter.get_neighborhood([triangle["a"]], depth=1, edge_types=["knows"])

    assert sorted(edge[2] for edge in edges) == ["knows", "trusts"]


async def test_get_neighborhood_of_an_absent_seed_is_empty(adapter, triangle):
    assert await adapter.get_neighborhood([str(uuid4())]) == ([], [])


async def test_get_nodeset_subgraph_or_takes_every_neighbour(adapter, triangle):
    nodes, edges = await adapter.get_nodeset_subgraph(_Ent, ["a", "c"])

    assert {properties["name"] for _, properties in nodes} == {"a", "b", "c"}
    assert len(edges) == 2


async def test_get_nodeset_subgraph_and_takes_only_shared_neighbours(adapter, triangle):
    """``b`` is adjacent to both named nodes; nothing else is."""
    nodes, _ = await adapter.get_nodeset_subgraph(_Ent, ["a", "c"], node_name_filter_operator="AND")
    assert {properties["name"] for _, properties in nodes} == {"a", "b", "c"}

    nodes, _ = await adapter.get_nodeset_subgraph(_Ent, ["a", "d"], node_name_filter_operator="AND")
    assert {properties["name"] for _, properties in nodes} == {"a", "d"}


async def test_get_nodeset_subgraph_of_an_unknown_name_is_empty(adapter, triangle):
    assert await adapter.get_nodeset_subgraph(_Ent, ["nobody"]) == ([], [])


async def test_get_nodeset_subgraph_of_another_type_is_empty(adapter, triangle):
    assert await adapter.get_nodeset_subgraph(_Other, ["a"]) == ([], [])


async def test_get_triplets_batch_pages(adapter, triangle):
    first = await adapter.get_triplets_batch(offset=0, limit=1)
    everything = await adapter.get_triplets_batch(offset=0, limit=10)

    assert len(first) == 1
    assert len(everything) == 2
    assert first[0]["start_node"]["name"] in {"a", "b"}
    assert "edge_text" in first[0]["relationship_properties"]


async def test_add_node_accepts_the_bare_id_form_the_interface_declares(adapter):
    await adapter.add_node("plain-id", {"name": "Plain"})
    assert (await adapter.get_node("plain-id"))["name"] == "Plain"


async def test_delete_node_and_delete_nodes_detach(adapter, triangle):
    await adapter.delete_node(triangle["a"])
    assert await adapter.has_node(triangle["a"]) is False
    assert await adapter.get_edges(triangle["b"]) == [
        (triangle["b"], triangle["c"], {"relationship_name": "likes"})
    ]

    await adapter.delete_nodes([triangle["b"], triangle["c"], "ghost"])
    assert await adapter.get_nodes([triangle["b"], triangle["c"]]) == []
