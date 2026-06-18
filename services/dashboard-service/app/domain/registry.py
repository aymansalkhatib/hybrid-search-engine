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
    port: int                # the service's port (internal, or host-published if external)
    external: bool = False   # True ⇒ published to the host (deep-link directly to :port)

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
            # Internal services aren't reachable on the host — the front-end deep-links
            # their Swagger through the gateway's grouped docs instead of :port.
            "external": self.external,
        }


def build_registry(settings) -> list[ServiceEntry]:
    """Assemble the registry from settings (so it stays config-driven)."""
    s = settings
    return [
        ServiceEntry("gateway", "API Gateway",
                     "Single entry point — the product UI talks only here",
                     "built", "gateway", s.api_gateway_url, s.api_gateway_port,
                     external=True),
        ServiceEntry("preprocessing", "Preprocessing",
                     "Normalize · tokenize · stopwords · stem · lemmatize",
                     "built", "pipeline", s.preprocessing_url, s.preprocessing_port),
        ServiceEntry("indexing", "Indexing",
                     "Inverted index (df · tf · avgdl) · Boolean match",
                     "built", "pipeline", s.indexing_url, s.indexing_port),
        ServiceEntry("docstore", "Doc Store",
                     "Docs + queries + qrels in MongoDB — read BY ID at query time",
                     "built", "pipeline", s.doc_store_url, s.doc_store_port),
        ServiceEntry("representation", "Representation",
                     "TF-IDF (VSM) · BM25 & Embeddings next",
                     "built", "pipeline", s.representation_url, s.representation_port),
        ServiceEntry("retrieval", "Retrieval",
                     "Match & rank · Hybrid + Fusion · Boolean (index-only)",
                     "built", "query", s.retrieval_url, s.retrieval_port),
        ServiceEntry("query-refinement", "Query Refinement",
                     "Spell-correct · expand · suggest",
                     "planned", "query", s.query_refinement_url, s.query_refinement_port),
        ServiceEntry("evaluation", "Evaluation",
                     "MAP · nDCG · Recall · P@10",
                     "planned", "query", s.evaluation_url, s.evaluation_port),
        ServiceEntry("mongo-express", "Mongo Express",
                     "Browse the document database (admin UI)",
                     "infra", "infra", None, s.mongo_express_port,
                     external=True),
    ]
