"""A provenance-complete Apache AGE graph adapter for cognee.

Importing this package does NOT register the adapter. Import
``cognee_age_adapter.register`` for that, so registration is an explicit act
and not an import side effect.
"""

from .adapter import AGEAdapter
from .constants import EDGE_LABEL, NODE_LABEL, PROVENANCE_COLUMNS, RELATIONSHIP_PROPERTY

__all__ = ["AGEAdapter", "EDGE_LABEL", "NODE_LABEL", "PROVENANCE_COLUMNS", "RELATIONSHIP_PROPERTY"]
__version__ = "0.1.0"
