"""Make one SQL statement from one Cypher query, as AGE requires.

AGE runs Cypher only inside SQL::

    SELECT * FROM ag_catalog.cypher('<graph>', $cypher$ <query> $cypher$, %s)
        AS (c0 ag_catalog.agtype, c1 ag_catalog.agtype, ...)

The column list must have one column for each item of the last ``RETURN``, and
a query without ``RETURN`` still needs one column. The third argument is the
one agtype map of the query parameters; AGE requires it to be a bound
parameter, which psycopg 3 sends as ``$1``.

Measured on 2026-11-08 against AGE 1.8.0, and each rule below has a test:

* AGE refuses a trailing ``;`` (``syntax error at or near ";"``).
* psycopg reads ``%`` as a placeholder, so the query text doubles it.
* The query text sits in a dollar-quoted string, so it must not hold the tag.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

AGTYPE = "ag_catalog.agtype"
DOLLAR_TAG = "$cypher$"

# AGE refuses a graph name shorter than 3 characters. The name goes into the
# SQL text, so anything other than a plain identifier is refused here.
_GRAPH_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,62}")

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_OPEN = "([{"
_CLOSE = ")]}"


def validate_graph_name(name: str) -> str:
    """Return ``name``, or raise ValueError if it cannot be an AGE graph name."""
    if not isinstance(name, str) or not _GRAPH_NAME.fullmatch(name):
        raise ValueError(
            f"invalid AGE graph name {name!r}: the graph name must be an identifier "
            "of 3 to 63 letters, digits or underscores that does not start with a digit"
        )
    return name


def compose(graph_name: str, query: str, columns: Optional[List[str]] = None) -> str:
    """The SQL statement that runs ``query`` on ``graph_name``.

    ``columns`` is the result of :func:`return_columns` when the caller has it.
    The SQL column names are ``c0``, ``c1``, ...: the Cypher names can be any
    text, and the caller maps the positions back to them.
    """
    validate_graph_name(graph_name)
    body = _strip_terminator(query)
    if DOLLAR_TAG in body:
        raise ValueError(f"the query text holds the dollar-quote tag {DOLLAR_TAG}")
    if columns is None:
        columns = return_columns(body)
    count = max(1, len(columns))
    column_list = ", ".join(f"c{index} {AGTYPE}" for index in range(count))
    return (
        f"SELECT * FROM ag_catalog.cypher('{graph_name}', "
        f"{DOLLAR_TAG}{body.replace('%', '%%')}{DOLLAR_TAG}, %s) AS ({column_list})"
    )


def return_columns(query: str) -> List[str]:
    """The names of the items of the last top-level ``RETURN`` of ``query``.

    An item with ``AS`` gives its alias; an item without one gives its own text,
    as neo4j names such a column. A query without ``RETURN`` gives ``[]``.
    ``RETURN *`` raises ValueError: AGE needs the column list before the query
    runs, and ``*`` names no column.
    """
    text = _strip_terminator(query)
    words, commas = _top_level_tokens(text)

    returns = [index for index, (_, _, word) in enumerate(words) if word == "RETURN"]
    if not returns:
        return []
    first = returns[-1]
    start = words[first][1]

    end = len(text)
    for index in range(first + 1, len(words)):
        word = words[index][2]
        is_order_by = word == "ORDER" and index + 1 < len(words) and words[index + 1][2] == "BY"
        if is_order_by or word in ("SKIP", "LIMIT"):
            end = words[index][0]
            break

    bounds = [start, *(comma for comma in commas if start < comma < end), end]
    columns = []
    for item_index, (item_start, item_end) in enumerate(zip(bounds, bounds[1:])):
        item_start += 0 if item_index == 0 else 1  # skip the comma
        item_words = [w for w in words if item_start <= w[0] < item_end]
        if item_index == 0 and item_words and item_words[0][2] == "DISTINCT":
            item_start = item_words[0][1]
            item_words = item_words[1:]
        aliases = [w for w in item_words if w[2] == "AS"]
        if aliases:
            columns.append(_unquote(text[aliases[-1][1] : item_end].strip()))
        else:
            columns.append(" ".join(text[item_start:item_end].split()))

    if columns == ["*"]:
        raise ValueError("RETURN * cannot be run on AGE: name each returned item")
    return columns


def _strip_terminator(query: str) -> str:
    return query.strip().rstrip(";").strip()


def _unquote(identifier: str) -> str:
    if len(identifier) >= 2 and identifier[0] == identifier[-1] == "`":
        return identifier[1:-1].replace("``", "`")
    return identifier


def _top_level_tokens(text: str) -> Tuple[List[Tuple[int, int, str]], List[int]]:
    """Words and commas outside strings, comments and brackets.

    Returns ``([(start, end, WORD_IN_UPPER_CASE)], [comma_position])``. A word
    after ``.`` (a property) or ``$`` (a parameter) is not a keyword and is not
    returned.
    """
    words: List[Tuple[int, int, str]] = []
    commas: List[int] = []
    depth = 0
    position = 0
    length = len(text)
    while position < length:
        char = text[position]
        if char in ("'", '"'):
            position = _skip_string(text, position)
            continue
        if char == "`":
            position = _skip_backticks(text, position)
            continue
        if text.startswith("//", position):
            newline = text.find("\n", position)
            position = length if newline < 0 else newline
            continue
        if text.startswith("/*", position):
            close = text.find("*/", position + 2)
            position = length if close < 0 else close + 2
            continue
        if char in _OPEN:
            depth += 1
        elif char in _CLOSE:
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            commas.append(position)
        else:
            word = _WORD.match(text, position)
            if word:
                previous = text[position - 1] if position else ""
                if depth == 0 and previous not in (".", "$") and not previous.isalnum():
                    words.append((word.start(), word.end(), word.group().upper()))
                position = word.end()
                continue
        position += 1
    return words, commas


def _skip_string(text: str, position: int) -> int:
    quote = text[position]
    position += 1
    while position < len(text):
        if text[position] == "\\":
            position += 2
            continue
        if text[position] == quote:
            return position + 1
        position += 1
    return position


def _skip_backticks(text: str, position: int) -> int:
    position += 1
    while position < len(text):
        if text[position] == "`":
            if text.startswith("``", position):
                position += 2
                continue
            return position + 1
        position += 1
    return position
