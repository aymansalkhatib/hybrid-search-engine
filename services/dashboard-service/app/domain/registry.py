"""The service registry — the single source of truth for what the console knows.

Each entry pairs a service's **internal** URL (the server-side proxy target on the
compose network) with its **host** port (so the browser can deep-link to its Swagger
or admin UI). ``group`` and ``tier`` drive how the front-end styles the node:

* ``group``: ``built`` (implemented), ``planned`` (not built yet — "offline" is
  expected, not an error), ``infra`` (a backing image / admin UI, not a proxy target).
* ``tier``: where the node sits in the SOA flow diagram.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ServiceEntry:
    key: str                 # used in /api/<key>/… and by the front-end
    label: str
    role: str
    group: str               # built | planned | infra
    tier: str                # gateway | pipeline | query | infra
    url: Optional[str]       # internal proxy target; None ⇒ browser-link only
    port: int                # host-published port (Swagger / admin UI links)

    def public(self) -> dict:
        """Browser-safe view (never leaks the internal compose URL)."""
        return {
            "key": self.key,
            "label": self.label,
            "role": self.role,
            "group": self.group,
            "tier": self.tier,
            "port": self.port,
            "proxied": self.url is not None,
        }


def build_registry(settings) -> list[ServiceEntry]:
    """Assemble the registry from settings (so it stays config-driven)."""
    s = settings
    return [
        ServiceEntry("gateway", "API Gateway",
                     "Single entry point — the product UI talks only here",
                     "built", "gateway", s.api_gateway_url, s.api_gateway_port),
        ServiceEntry("preprocessing", "Preprocessing",
                     "Normalize · tokenize · stopwords · stem · lemmatize",
                     "built", "pipeline", s.preprocessing_url, s.preprocessing_port),
        ServiceEntry("indexing", "Indexing",
                     "Build & serve the inverted index (df · tf · avgdl)",
                     "built", "pipeline", s.indexing_url, s.indexing_port),
        ServiceEntry("docstore", "Doc Store",
                     "Raw docs in MongoDB — read BY ID at query time",
                     "built", "pipeline", s.doc_store_url, s.doc_store_port),
        ServiceEntry("representation", "Representation",
                     "TF-IDF · BM25 · Embeddings",
                     "planned", "query", s.representation_url, s.representation_port),
        ServiceEntry("retrieval", "Retrieval",
                     "Match & rank · Hybrid Serial/Parallel + Fusion",
                     "planned", "query", s.retrieval_url, s.retrieval_port),
        ServiceEntry("query-refinement", "Query Refinement",
                     "Spell-correct · expand · suggest",
                     "planned", "query", s.query_refinement_url, s.query_refinement_port),
        ServiceEntry("evaluation", "Evaluation",
                     "MAP · nDCG · Recall · P@10",
                     "planned", "query", s.evaluation_url, s.evaluation_port),
        ServiceEntry("mongo-express", "Mongo Express",
                     "Browse the document database (admin UI)",
                     "infra", "infra", None, s.mongo_express_port),
    ]
