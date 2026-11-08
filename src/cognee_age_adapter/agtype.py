"""Read the text form of an agtype value into plain Python.

psycopg returns agtype as text. The text is JSON with type annotations after
some values, outside strings::

    {"id": 844424930131969, "label": "Node", "properties": {...}}::vertex
    {"id": ..., "label": "Edge", "start_id": ..., "end_id": ..., "properties": {...}}::edge
    [<vertex>, <edge>, <vertex>]::path
    10::numeric

A vertex or an edge becomes its property map, as neo4j's ``Result.data()``
returns it and as the FalkorDB adapter did. A path becomes the list of its
elements. ``::numeric`` becomes a Python number.

⚠ A cast to ``json`` in SQL is not an alternative: measured on AGE 1.8.0, a
vertex cast to json fails with ``agtype_value_to_text: unsupported argument
agtype 6``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

# A string literal is matched first, so an annotation inside a string stays text.
_ANNOTATION = re.compile(
    r'"(?:[^"\\]|\\.)*"'
    r"|(?P<entity>\}::(?:vertex|edge))"
    r"|(?P<path>\]::path)"
    r"|(?P<number>::(?:numeric|float|integer|int))"
)

# The marker key that a vertex or an edge gets during the parse. agtype text
# cannot hold NUL, so no stored key can be equal to it.
_MARKER = "\x00agtype"
_MARKER_JSON = json.dumps(_MARKER)


def _annotate(match: "re.Match[str]") -> str:
    if match.group("entity"):
        return f", {_MARKER_JSON}: true}}"
    if match.group("path"):
        return "]"
    if match.group("number"):
        return ""
    return match.group(0)


def _unwrap(value: dict) -> Any:
    if _MARKER in value:
        return value.get("properties") or {}
    return value


def parse(text: Optional[str]) -> Any:
    """Return the Python value of one agtype result value. SQL NULL is None."""
    if text is None:
        return None
    if "::" not in text:
        return json.loads(text)
    return json.loads(_ANNOTATION.sub(_annotate, text), object_hook=_unwrap)
