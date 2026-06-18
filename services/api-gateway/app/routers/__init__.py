"""Gateway routers — one module per concern.

* ``catalog``        — dataset options + live status the UI renders (aggregations).
* ``lifecycle``      — the offline dataset flow (download → ingest → index) + jobs.
* ``preprocessing``  — passthrough to the preprocessing-service.
* ``indexing``       — passthrough to the indexing-service.
* ``representation`` — passthrough to the representation-service (incl. the online search).
* ``docstore``       — passthrough to the doc-store read surface (by-id / browse).

Each per-service router mirrors that service's own API under a ``/<service>`` prefix
and reuses the shared ``shared.contracts`` models, so the gateway's Swagger documents
the whole system and validates input — the services themselves are not published to
the host (compose network only).
"""
