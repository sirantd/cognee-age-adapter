"""initialize() makes the graph, its labels and its indexes once per adapter instance.

cognee's ``get_graph_engine()`` calls ``initialize()`` on every engine lookup, so
a repeated call must not reach PostgreSQL again. A second instance (a second
process) sends every create again, and each one that exists is not an error.
"""

from __future__ import annotations

import pytest

import conftest

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def test_a_repeat_call_sends_nothing(adapter):
    sent = conftest.record_sql(adapter)
    await adapter.initialize()
    await adapter.initialize()
    assert sent == []


async def test_a_second_instance_on_the_same_graph_initializes(adapter):
    second = conftest.new_adapter(adapter._graph_name)
    sent = conftest.record_sql(second)
    try:
        await second.initialize()
    finally:
        await second.close()
    assert any("create_graph" in sql for sql, _ in sent)
    assert any("CREATE UNIQUE INDEX IF NOT EXISTS" in sql for sql, _ in sent)


async def test_labels_are_created_one_statement_at_a_time(adapter):
    """🚨 Measured on AGE 1.8.0: 3,000 labels created in one transaction failed
    with ``out of shared memory``. Each create is its own statement here."""
    second = conftest.new_adapter(adapter._graph_name)
    sent = conftest.record_sql(second)
    try:
        await second.initialize()
    finally:
        await second.close()
    creates = [sql for sql, _ in sent if "create_vlabel" in sql or "create_elabel" in sql]
    assert len(creates) == 3
    assert all(sql.count("create_") == 1 for sql in creates)
