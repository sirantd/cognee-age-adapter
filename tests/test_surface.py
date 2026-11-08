"""The surface gate: the adapter covers its target surface, with signatures that match.

Deliberately needs NO AGE server. Whether every abstract method is present
is a property of the class object (``__abstractmethods__``), and signature drift
is a property of two ``inspect.Signature`` objects — turning either into an
integration test would mean the cheapest, highest-value check only runs when
infrastructure happens to be up.

The signature check is the one that earns its keep. cognee calls most of these
from inside a pipeline, so a drifted parameter surfaces as a TypeError three
layers from the cause, during a drain, at 03:00.
"""

from __future__ import annotations

import inspect

import pytest

from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface

from cognee_age_adapter import AGEAdapter

# Reached by cognee's runtime, so the adapter must implement them even though the
# interface supplies a raising default. Derived in the spec by grepping every
# call site outside the adapter implementations themselves.
RUNTIME_REACHED = frozenset(
    """
    attach_edge_source_refs attach_node_source_refs delete_edge_triples
    find_edge_source_refs_by_dataset find_edge_source_refs_by_pipeline_run
    find_edges_by_source_ref find_node_source_refs_by_dataset
    find_node_source_refs_by_pipeline_run find_nodes_by_source_ref
    get_edge_delete_data get_graph_metadata get_node_delete_data
    get_node_truth_state get_triplets_batch remove_edge_source_refs
    remove_node_source_refs set_graph_metadata set_node_truth_state
    """.split()
)

# ⚠ Scoped as "override only if slow", and that was wrong. The interface's
# default is a NO-OP, while the provenance contract suite asserts the tags are
# actually removed — so a backend that inherits it fails
# `test_remove_belongs_to_set_tags_scoped_and_unscoped`. Both in-core adapters
# that pass the suite implement it. Folded into the target here so the
# correction has a test behind it: the build target is 41 methods, not 40.
MUST_OVERRIDE_A_REAL_DEFAULT = frozenset({"remove_belongs_to_set_tags"})

# Left inheriting the raising default on purpose. Asserted explicitly so that "we
# skipped these" stays a decision with a test behind it rather than an oversight
# nobody notices — which is the whole reason this set is a closed list and not a
# comment.
#
# The eight weight methods are reached by NOTHING in cognee's runtime.
#
# `update_node` (added to the interface in cognee 1.5.0) is the one entry with a
# caller: `cognee/tasks/storage/close_node.py`, the bi-temporal "stamp valid_to"
# helper. It is safe to leave raising because that caller CATCHES
# NotImplementedError, logs a warning and returns False — and because nothing on
# the cognify, recall or delete path calls close_node at all (only a docstring in
# DataPoint.py references it). Upstream declares it an "optional extension" for
# exactly this reason. 💡 If a caller ever lands on the cognify path, implement it
# as a read-modify-write MERGE and move it into RUNTIME_REACHED.
DELIBERATELY_UNIMPLEMENTED = frozenset(
    """
    get_edge_feedback_weights get_edge_frequency_weights get_node_feedback_weights
    get_node_frequency_weights set_edge_feedback_weights set_edge_frequency_weights
    set_node_feedback_weights set_node_frequency_weights
    update_node
    """.split()
)


def _abstract_methods() -> frozenset[str]:
    return frozenset(
        name
        for name, value in vars(GraphDBInterface).items()
        if getattr(value, "__isabstractmethod__", False)
    )


def test_adapter_is_concrete():
    """Every abstract method is implemented."""
    assert AGEAdapter.__abstractmethods__ == frozenset(), (
        "unimplemented abstract methods: "
        f"{sorted(AGEAdapter.__abstractmethods__)}"
    )


def _target() -> frozenset[str]:
    return _abstract_methods() | RUNTIME_REACHED | MUST_OVERRIDE_A_REAL_DEFAULT | {"has_node"}


def test_target_surface_is_defined_on_the_adapter():
    """The whole target is defined here, not inherited from the interface."""
    target = _target()
    own = set()
    for klass in AGEAdapter.__mro__:
        if klass is GraphDBInterface:
            break
        own |= set(vars(klass))

    missing = sorted(target - own)
    assert not missing, f"target methods still inherited from the interface: {missing}"


def test_weight_surface_is_left_inherited_on_purpose():
    """The deliberately-unimplemented methods must NOT be implemented — see the spec."""
    own = set()
    for klass in AGEAdapter.__mro__:
        if klass is GraphDBInterface:
            break
        own |= set(vars(klass))

    unexpected = sorted(DELIBERATELY_UNIMPLEMENTED & own)
    assert not unexpected, (
        "these are reached by nothing in cognee's runtime and were deliberately "
        f"left raising; implementing them needs a spec change first: {unexpected}"
    )


@pytest.mark.parametrize(
    "name", sorted(_abstract_methods() | RUNTIME_REACHED | MUST_OVERRIDE_A_REAL_DEFAULT)
)
def test_signature_does_not_narrow_the_interface(name):
    """Every call valid against GraphDBInterface must be valid against us.

    Not exact equality — cognee's own adapters do not match the interface
    literally, and requiring it would be a test of a convention nobody follows.
    Measured against 1.4.1 (not re-measured against the 1.5.2 pin):
    `query(self, query, params)` on the interface becomes
    `params=None` on Neo4j, ladybug AND Postgres, and Postgres additionally
    renames the first parameter to `query_str`.

    So the rule is widening-only:
      * same parameter names, order and kinds  — positional calls keep working
      * we may ADD a default where the interface has none  — widens acceptance
      * we may NOT drop a default the interface declares  — would break callers
      * we may NOT add a new required parameter            — same

    Annotations are excluded: this module uses ``from __future__ import
    annotations``, so ours are strings and theirs are types — a formatting
    difference, not drift.
    """
    ours = list(inspect.signature(getattr(AGEAdapter, name)).parameters.values())
    theirs = list(inspect.signature(getattr(GraphDBInterface, name)).parameters.values())

    assert [p.name for p in ours] == [p.name for p in theirs], (
        f"{name}: parameter names/order drifted\n"
        f"  ours:   {[p.name for p in ours]}\n  theirs: {[p.name for p in theirs]}"
    )
    assert [p.kind for p in ours] == [p.kind for p in theirs], f"{name}: parameter kinds drifted"

    for mine, mandated in zip(ours, theirs):
        if mandated.default is not inspect.Parameter.empty:
            assert mine.default is not inspect.Parameter.empty, (
                f"{name}: parameter {mine.name!r} drops the interface's default "
                f"{mandated.default!r} — that narrows acceptance and breaks callers "
                f"who omit it"
            )
