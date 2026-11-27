"""``find_edge_texts_in_use`` equals the interface default on texts the upstream case lacks.

cognee's contract case pads with spaces and a newline only. A false negative here
makes cognee delete the EdgeType entry of a text that an edge still has, without a
trace. So every padding that ``str.strip()`` removes, and every JSON type of
``edge_text``, is compared with ``GraphDBInterface.find_edge_texts_in_use`` (the
whole-graph read in Python).
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
from cognee.infrastructure.engine import DataPoint

from conftest import throwaway_adapter

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


class Entity(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"]}


EDGE_TEXTS = {
    "nbsp": "\u00a0nbsp padded\u00a0",
    "tab": "\ttab padded\t",
    "ideographic": "\u3000ideographic padded\u3000",
    "number": 5,
    "bool": True,
    "list": ["a", "b"],
    "blank_tab": "\t",
}

ASKED = [
    "nbsp padded",
    "tab padded",
    "ideographic padded",
    "5",
    "True",
    "true",
    "['a', 'b']",
    "blank_tab",
    "absent",
    "no edge has this",
]


@pytest.fixture
async def adapter():
    async with throwaway_adapter("edge_texts") as adapter:
        yield adapter


async def _seed(adapter) -> None:
    nodes = [Entity(id=uuid4(), name=str(i)) for i in range(2)]
    await adapter.add_nodes(nodes)
    a, b = str(nodes[0].id), str(nodes[1].id)
    edges = [(a, b, name, {"edge_text": value}) for name, value in EDGE_TEXTS.items()]
    edges.append((a, b, "absent", {}))
    await adapter.add_edges(edges)


async def test_native_equals_the_interface_default(adapter):
    await _seed(adapter)
    native = await adapter.find_edge_texts_in_use(ASKED)
    default = await GraphDBInterface.find_edge_texts_in_use(adapter, ASKED)
    assert native == default
    assert {"nbsp padded", "tab padded", "ideographic padded", "blank_tab", "absent"} <= native


async def test_more_texts_than_edges_reads_every_edge(adapter):
    await _seed(adapter)
    asked = ASKED + [f"filler {i}" for i in range(50)]
    native = await adapter.find_edge_texts_in_use(asked)
    assert native == await GraphDBInterface.find_edge_texts_in_use(adapter, asked)
