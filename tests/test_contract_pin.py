"""The other half of the contract gate: the pin, and the census that makes the pin mean something.

``test_contract.py`` proves the adapter satisfies cognee's provenance contract.
That proof is worth exactly as much as the cognee it ran against is fixed:
the graph interface has moved twice in four months, and an unpinned suite tests
whatever pip resolved that morning while still reporting 20 green.

So three things are asserted here, all server-free:

* the test extra pins cognee **exactly** (``==``, not a range),
* the installed cognee **is** that pin,
* the suite still holds the **same 20 cases**, by name, and the gate module
  really did shadow the fixture rather than inherit the suite's own providers.

📌 The census is the part that catches a *deliberate* upgrade. Bumping the pin
makes these tests fail before anything else does, which forces the suite to be
re-read rather than re-run — a renamed or deleted case would otherwise shrink
coverage silently, and a new one would arrive unnoticed.
"""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import tomllib
from pathlib import Path

from cognee.tests.integration.infrastructure.graph import (
    test_graph_provenance_adapter_contract as upstream,
)

import test_contract as gate

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

# The suite as of the pinned cognee. Frozen on purpose: this list is the
# difference between "20 tests passed" and "the 20 tests we signed off passed".
CONTRACT_TESTS = frozenset(
    """
    test_add_edges_folds_multiple_owners
    test_add_edges_folds_provenance_in_one_write
    test_add_nodes_folds_provenance_in_one_write
    test_attach_node_source_refs_materializes_all_fields
    test_attach_without_pipeline_run_is_not_rollbackable_by_run
    test_concurrent_explicit_attach_keeps_all_keys
    test_concurrent_folded_attach_keeps_all_keys
    test_concurrent_folded_writes_and_attaches_keep_every_owner
    test_delete_edge_triples_preserves_endpoints
    test_edge_provenance_snapshot_and_lookups
    test_folded_attach_omitted_when_no_source_ref
    test_folded_attach_preserves_model_a_on_node_and_edge_reattach
    test_graph_metadata_round_trip
    test_node_delete_data_snapshot_fields
    test_node_source_ref_lookups_are_exact
    test_remove_belongs_to_set_tags_scoped_and_unscoped
    test_remove_keeps_dataset_id_when_sibling_ref_shares_dataset
    test_remove_keeps_run_id_when_sibling_ref_shares_run
    test_remove_node_source_refs_updates_derived_and_is_idempotent
    test_rewrite_preserves_provenance
    """.split()
)


def _pinned_cognee_version() -> str:
    """The version in ``[project.optional-dependencies].test``, or an error.

    Fails on anything that is not a single exact pin, so relaxing the
    requirement to a range breaks this test rather than quietly widening what
    the gate runs against.
    """
    test_extra = tomllib.loads(PYPROJECT.read_text())["project"]["optional-dependencies"]["test"]
    pins = [
        requirement
        for requirement in test_extra
        if requirement.replace(" ", "").startswith("cognee==")
    ]

    assert len(pins) == 1, (
        "the test extra must pin cognee with exactly one `cognee==<version>` "
        f"requirement — the contract suite is only a drift detector if the cognee it "
        f"comes from is fixed. Found: {test_extra}"
    )
    return pins[0].split("==", 1)[1].strip()


def test_the_test_extra_pins_cognee_exactly():
    assert _pinned_cognee_version()


def test_the_installed_cognee_is_the_pinned_one():
    """Guards the environment, not the file — an editable install can drift."""
    assert importlib_metadata.version("cognee") == _pinned_cognee_version()


def test_the_contract_suite_is_the_one_we_signed_off():
    """🚨 20 cases, by name. A rename shrinks coverage without failing anything."""
    found = frozenset(name for name in vars(upstream) if name.startswith("test_"))

    assert len(CONTRACT_TESTS) == 20
    assert found == CONTRACT_TESTS, (
        "cognee's provenance contract suite has changed under the pin.\n"
        f"  added:   {sorted(found - CONTRACT_TESTS)}\n"
        f"  removed: {sorted(CONTRACT_TESTS - found)}\n"
        "Read the diff before updating this list — a new case is new required "
        "behaviour, and a removed one is coverage this adapter no longer has."
    )


def test_the_gate_module_collects_every_contract_test():
    """The star-import is load-bearing; assert it actually bound all 20."""
    missing = sorted(CONTRACT_TESTS - frozenset(vars(gate)))
    assert not missing, f"not collected by tests/test_contract.py: {missing}"

    for name in sorted(CONTRACT_TESTS):
        assert getattr(gate, name) is getattr(upstream, name), (
            f"{name} in the gate module is not cognee's — the suite must run "
            "unmodified, or it stops being cognee's contract"
        )


def test_the_gate_runs_age_and_only_age():
    """The whole mechanism is one shadowed fixture; prove the shadow took.

    Without the override the suite would run its own ladybug/postgres/neo4j
    params — ladybug is installed, so it would pass, and the gate would be green
    having tested a different backend entirely.

    📌 From cognee 1.6.6 the suite's fixture lives in the package's
    ``conftest.py``, not in the suite module. pytest loads no ``conftest.py``
    from ``site-packages``, so the gate's fixture is the only provider; the
    ``getattr`` keeps the check honest if the fixture ever moves back.
    """
    assert gate.PROVIDERS == ["age"]
    assert "graph_provenance_adapter" in vars(gate)
    assert gate.graph_provenance_adapter is not getattr(upstream, "graph_provenance_adapter", None)
