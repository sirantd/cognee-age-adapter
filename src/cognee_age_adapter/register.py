"""Register the adapter with cognee as a GRAPH provider, and only a graph one.

🚨 This module must never bind a vector adapter. The deployment's embeddings
live in cognee's pgvector tables, with the column width LOCKED at 1024 by
``baai/bge-m3``. Moving them is a re-embedding of the whole corpus, not a
configuration change. homelab#332 keeps them where they are.

Importing this module is the whole registration. It is idempotent.
"""

from __future__ import annotations

import os

from cognee.infrastructure.databases.graph.use_graph_adapter import use_graph_adapter

from .adapter import AGEAdapter

# The value of GRAPH_DATABASE_PROVIDER that selects this adapter.
PROVIDER_NAMES = ("age",)


def ensure_registered(assert_vector_provider: bool = True) -> None:
    """Bind AGEAdapter under its provider name.

    ``assert_vector_provider`` fails loudly if the VECTOR store is set to
    anything other than pgvector. By the time cognee runs, it is too late to
    notice this cheaply.
    """
    for name in PROVIDER_NAMES:
        use_graph_adapter(name, AGEAdapter)

    if assert_vector_provider:
        vector_provider = os.getenv("VECTOR_DB_PROVIDER", "").strip().lower()
        if vector_provider and vector_provider not in ("pgvector", "postgres"):
            raise RuntimeError(
                "cognee-age-adapter is a GRAPH adapter only, but "
                f"VECTOR_DB_PROVIDER is {vector_provider!r}. Embeddings must stay "
                "on pgvector: this deployment's vector column is locked at 1024 "
                "dimensions by baai/bge-m3."
            )


ensure_registered()
