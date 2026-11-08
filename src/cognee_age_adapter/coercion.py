"""The value rules of the FalkorDB adapter, kept for AGE on purpose.

Two jobs, kept apart because they fail differently:

* :func:`coerce_properties` — **shape**. The form in which a value reaches the
  store.
* :func:`scrub_nul` — **transport**. What a parameter can carry. It runs on
  the parameters of every statement (``AGEAdapter._cypher``), so it also covers
  the paths that never build a property map: tag lists, filter values, graph
  metadata and ids.

📌 **The rules are the rules of cognee-falkordb-adapter 0.5.4, unchanged.**
homelab#332 (story 28) requires that NUL bytes and odd model output reach the
store in the same form as on FalkorDB. Then a node that the migration copies
from FalkorDB and the same node that a re-cognify writes on AGE are equal, and
a re-cognify does not rewrite the graph.

AGE can hold more than FalkorDB could. Measured on 2026-11-08 against AGE 1.8.0
(``apache/age:release_PG18_1.8.0``) with psycopg 3.3:

| input | AGE behaviour | rule (unchanged) | why the rule stays |
|---|---|---|---|
| ``None`` property value | ``SET n += {k: null}`` **DELETES** ``k`` | drop the key | load-bearing |
| ``None`` inside an array | stored natively | drop the entry | same form as FalkorDB |
| map | stored natively | JSON-encode | same form as FalkorDB |
| array that holds a map | stored natively | JSON-encode | same form as FalkorDB |
| nested array of primitives | stored natively | pass through | — |
| ``UUID``, ``bytes``, other types | not JSON, the parameter map cannot carry it | ``str()``, ``decode()`` for bytes | load-bearing |
| ``\\x00`` anywhere | ``UntranslatableCharacter``: whole statement | strip | load-bearing |
| every other C0, and DEL | accepted, round-trips | keep | — |

🚨 A null is a **delete**, not a no-op. A bare cognee DataPoint dumps 7
null-valued keys, so a pass-through would make every re-cognify remove
``source_pipeline``, ``source_node_set`` and the other keys from a node that
another pipeline had filled in.

⚠ **There is no "clear this property" path through this layer, by design.** A
``None`` means "this DataPoint does not carry the field". It does not mean
"delete what is stored". cognee's ladybug adapter takes the same position.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional
from uuid import UUID

from cognee.modules.storage.utils import JSONEncoder
from cognee.shared.logging_utils import get_logger

logger = get_logger("AGEAdapter")

# The one character that PostgreSQL text cannot hold. A parameter map with
# "\u0000" fails the statement with UntranslatableCharacter.
NUL = "\x00"

# Types that the JSON parameter map carries as they are. Anything else becomes
# its string form here.
_STORABLE_SCALARS = (str, int, float, bool)


def coerce_properties(properties: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Return ``properties`` with every value in the form the store keeps.

    Null-valued keys are **absent** from the result, not null in it: the two are
    different writes. ``SET n += {k: null}`` removes a stored ``k``; an absent key
    leaves it alone.
    """
    coerced: Dict[str, Any] = {}
    dropped: List[str] = []

    for key, value in (properties or {}).items():
        if value is None:
            dropped.append(key)
            continue
        coerced[key] = coerce_value(value)

    if dropped:
        # Debug, not warning: unset optionals are the normal case on every
        # DataPoint, so this is a trace of what was not written, not an anomaly.
        logger.debug("Dropped %d null-valued propert(ies): %s", len(dropped), ", ".join(dropped))

    return coerced


def coerce_value(value: Any) -> Any:
    """Coerce one property value. See the module docstring for the rule table.

    📌 Public: the Stage D loader writes through its own Cypher, and it must
    apply the same rules to the values it loads.
    """
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, bytes):
        # Decoded, not str()'d: `str(b"hi")` stores the repr `b'hi'`.
        return value.decode(errors="replace")
    if isinstance(value, dict):
        return json.dumps(value, cls=JSONEncoder)
    if isinstance(value, (list, tuple)):
        if _contains_map(value):
            # This goes out as JSON, so nulls inside it are representable and
            # are kept. Only a native array drops them.
            return json.dumps(list(value), cls=JSONEncoder)
        return _storable_array(value)
    if isinstance(value, _STORABLE_SCALARS):
        return value
    return str(value)


def _contains_map(value: Any) -> bool:
    """True if a map is anywhere inside ``value``."""
    if isinstance(value, dict):
        return True
    if isinstance(value, (list, tuple)):
        return any(_contains_map(item) for item in value)
    return False


def _storable_array(values: Any) -> List[Any]:
    """A native array: nulls removed, nested arrays kept, scalars coerced."""
    storable: List[Any] = []
    for item in values:
        if item is None:
            continue
        if isinstance(item, (list, tuple)):
            storable.append(_storable_array(item))
        else:
            storable.append(coerce_value(item))
    return storable


def scrub_nul(value: Any) -> Any:
    """Recursively remove NUL from strings, container values and mapping keys.

    Returns the **original object** when there is nothing to scrub. Every
    query's parameters pass through here, also 1,000-node write batches, and the
    common path must not rebuild the whole structure to change nothing. ⚠ A
    tuple that did need scrubbing comes back as a list; JSON encodes the two
    identically.
    """
    return _scrub(value) if _has_nul(value) else value


def _has_nul(value: Any) -> bool:
    if isinstance(value, str):
        return NUL in value
    if isinstance(value, dict):
        return any(_has_nul(key) or _has_nul(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_has_nul(item) for item in value)
    return False


def _scrub(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace(NUL, "")
    if isinstance(value, dict):
        return {_scrub(key): _scrub(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub(item) for item in value]
    return value
