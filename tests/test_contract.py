"""The gate: cognee's own graph-provenance contract suite, on AGE.

The suite is cognee's, not ours, and it is backend-neutral by construction. Its
own docstring invites this:

    "Add a provider to ``graph_provenance_adapter`` when implementing graph
    provenance for a new graph backend."

📌 **We add that provider by importing the suite and shadowing one fixture, not
by editing cognee.** The suite ships inside the cognee wheel, so an edited copy
would live in ``site-packages``: unversioned, silently reverted by the next
``pip install``, and invisible in review. The star-import below binds the same
20 test functions into this module, where pytest collects them and resolves
``graph_provenance_adapter`` against the definition here.

⚠ **The suite is a gate only while the cognee it came from is pinned.**
``pyproject.toml`` pins it exactly and ``test_contract_pin.py`` enforces the pin
and the 20 names, so drift fails here and not in the 03:00 drain.
"""

from __future__ import annotations

import pytest

# The suite itself. The star-import puts the 20 test functions in this module's
# namespace for pytest to collect, and lets the fixture below shadow the
# suite's fixture by name.
from cognee.tests.integration.infrastructure.graph.test_graph_provenance_adapter_contract import *  # noqa: F401,F403

from conftest import throwaway_adapter

# Overrides the ``pytestmark`` that the star-import brought along (asyncio only).
# These need a server, and the marker keeps the server-free CI job server-free.
pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

# What cognee's fixture calls its ``params``. One entry, because the other
# providers are cognee's to run. ``test_contract_pin.py`` reads it and asserts
# that the override took instead of the suite's own ladybug/postgres/neo4j.
PROVIDERS = ["age"]


@pytest.fixture(params=PROVIDERS)
async def graph_provenance_adapter(request):
    """Shadows the suite's fixture of the same name; yields an AGE adapter.

    A throwaway graph for each case, so every test starts from an empty store,
    as the suite's ladybug (fresh ``tmp_path``) and neo4j (``DETACH DELETE``)
    branches do. An unreachable server skips, or fails under ``AGE_REQUIRED``.
    """
    assert request.param == "age", f"unexpected provider: {request.param}"
    async with throwaway_adapter("contract") as adapter:
        yield adapter
