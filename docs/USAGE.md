# Usage Guide — how to use what's built so far

This guide explains **every service that exists today** and how to drive it, with copy‑paste
examples and the actual responses you should see. It grows as new services land
(representations, retrieval, UI come later).

> **Implemented now:** `preprocessing-service`, `indexing-service`, `doc-store-service`
> (+ MongoDB) and the `api-gateway` (now with a **dataset catalog** + thin action proxies).

---

## 1. The mental model: offline build vs. online query

The system is split into two phases (a hard requirement — the online path must answer in
**≤ ~20 s** with **no training at query time**):

| Phase | When | What happens | Can be slow? |
|-------|------|--------------|--------------|
| **Offline** | once, up front | **download** dataset → **ingest** raw docs to Mongo (migration) → **build inverted index from Mongo** (and later: TF‑IDF/embeddings/BM25) | ✅ yes |
| **Online** | per user query | preprocess the query → score with the prebuilt models → **fetch the top‑k original docs by id from Mongo** | ❌ must be fast |

The pipeline is strictly **download → ingest → build**: MongoDB is the **single source of
truth** for documents. The indexer reads the corpus from the doc‑store (not from the dataset
cache), so the index is built from the **same** docs the UI displays — and you can **browse**
the stored corpus directly.

Because the offline steps are slow, **download / ingest / build run as background jobs**: the
endpoint returns a small **`JobRef`** immediately and you poll its **progress (percent)** until
it finishes. This is exactly what a UI needs to show a progress bar.

What you can already do today: **see a dataset catalog**, **download a dataset (with progress)**,
**store raw docs and read them by id**, and **build & query an inverted index (whole or a range)**.

---

## 2. Start the services

```bash
cp .env.example .env          # one-time: configure datasets, ports, models
docker compose up -d api-gateway preprocessing-service indexing-service doc-store-service doc-store-db
docker compose ps             # all should be "healthy"/"running"
```

Run the **whole** system with `docker compose up`, or a **single** service with
`docker compose up <service>`.

| Service | Port | Swagger UI | Purpose |
|---------|------|-----------|---------|
| api-gateway | 8000 | http://localhost:8000/docs | single entry point — **catalog** + action proxies |
| preprocessing-service | 8001 | http://localhost:8001/docs | normalize/tokenize text |
| indexing-service | 8002 | http://localhost:8002/docs | inverted index (postings, df, doc lengths) |
| doc-store-service | 8007 | http://localhost:8007/docs | **raw** docs in MongoDB, read by id |
| doc-store-db (MongoDB) | 27017 | — | document database (engine) |

Every service answers `GET /health` and `GET /` (info).

> **PowerShell note:** in PowerShell `curl` is an alias for `Invoke-WebRequest`. Use
> **`curl.exe`** for the examples below, or use `Invoke-RestMethod` (shown where it helps).

---

## 3. The async-job pattern (read once, applies to download/ingest/build)

Every offline action returns a **`JobStatus`** and runs in the background:

```jsonc
// POST .../download  ->  202 Accepted
{ "job_id": "a1b2c3…", "type": "download", "key": "beir/quora/test",
  "state": "running", "processed": 0, "total": 522931, "percent": 0.0, "message": "…" }
```

Poll the job until `state` is terminal (`succeeded` / `skipped` / `failed`):

```bash
curl -s localhost:8000/jobs/docstore/a1b2c3…     # build jobs live under .../indexing/<id>
# running:   {"state":"running","processed":120000,"total":522931,"percent":22.95,...}
# done:      {"state":"succeeded","percent":100.0,"result":{"doc_count":522931}}
```

- `percent` is `null` until the total is known (e.g. while the corpus **archive** is downloading,
  before per‑document counting starts).
- Re‑triggering the same action while a job is in flight returns **409**.
- An already‑done action returns a job in state **`skipped`** (no work, no network).

---

## 4. End‑to‑end walkthrough (Dataset A = Quora)

A dataset is referenced by its **id** — a value from the `DATASETS` catalog in `.env` (the first
entry is the primary/required one). The examples below use `beir/quora/test` and go through the
gateway at **:8000** (each service is also reachable directly — ports above).

### Step 0 — See the catalog (the "options" for a UI)

```bash
curl -s localhost:8000/catalog
# -> {"datasets":[
#      {"dataset_id":"beir/quora/test",
#       "download":{"downloaded":false,"doc_count":null},
#       "ingest":{"ingested_count":0,"fully_ingested":null},
#       "index":{"built":false,"num_docs":null},"active_jobs":[]},
#      {"dataset_id":"wikir/en1k/test", ...}]}
```

The catalog lists **only** the datasets in `DATASETS` (`.env`) — it never searches the internet,
and there is **no default**: if `DATASETS` is unset the catalog is empty and dataset actions
report *"no datasets configured"*.

### Step 0.5 — Preview a dataset's details (before downloading it)

```bash
curl -s "localhost:8000/datasets/info?dataset=beir/quora/test"
# -> {"dataset_id":"beir/quora/test","doc_count":522931,"num_queries":10000,
#     "num_qrels":15675,"has_qrels":true,"downloaded":false}
```

Network‑free (from ir‑datasets metadata) — so a UI can show a dataset's size and whether it has
qrels **before** the user commits to downloading it.

### Step 1 — Download the dataset locally (the ONLY step that touches the internet)

```bash
curl -s -X POST localhost:8000/datasets/download \
  -H "Content-Type: application/json" -d '{"dataset":"beir/quora/test"}'   # add "force":true to re-fetch
# -> {"job_id":"…","type":"download","state":"running","total":522931,"percent":0.0,...}
curl -s localhost:8000/jobs/docstore/<job_id>               # poll until "succeeded"
```

The corpus is materialized once into `data/dataset/beir/quora/test/` and marked complete with a
`.manifest.json`. Everything below runs **offline**; if you skip this, ingest/build return **409**.

### Step 2 — Ingest (migrate) the raw docs into Mongo (offline, async)

This is the **migration** that makes MongoDB the source of truth; the index is built from it.

```bash
curl -s -X POST localhost:8000/datasets/ingest \
  -H "Content-Type: application/json" -d '{"dataset":"beir/quora/test","limit":500}'  # omit limit for all
# -> JobRef; poll localhost:8000/jobs/docstore/<id> -> "succeeded" {"ingested_count":500}
```

You can **browse** what was stored right away (also powers a future UI database viewer):

```bash
curl -s "localhost:8000/datasets/docs?dataset=beir/quora/test&offset=0&limit=2"
# -> {"dataset_id":"beir/quora/test","total":500,"offset":0,"limit":2,
#     "docs":[{"seq":0,"doc_id":"1","text":"What is the step ..."}, {"seq":1,...}]}
```

### Step 3 — Build the inverted index **from MongoDB** (offline, async — whole or a range)

Reads the corpus from the doc‑store, so it **requires Step 2 first** (else **409**).

```bash
# a positional RANGE [start:stop) over the stored `seq` order:
curl -s -X POST localhost:8000/datasets/index \
  -H "Content-Type: application/json" -d '{"dataset":"beir/quora/test","start":0,"stop":500}'
# whole stored corpus: -d '{"dataset":"beir/quora/test"}'   |   smoke cap: add "limit":500
# -> JobRef; poll localhost:8000/jobs/indexing/<id> -> "succeeded" {"stats":{...}}
```

### Step 4 — Inspect / query the index (direct to indexing :8002)

```bash
curl -s "localhost:8002/index/status?dataset=beir/quora/test"   # {built, stats, active_job} — live progress
curl -s "localhost:8002/stats?dataset=beir/quora/test"
curl -s "localhost:8002/term?dataset=beir/quora/test&term=india"      # -> {"term":"india","df":13,"cf":13}
curl -s "localhost:8002/postings?dataset=beir/quora/test&term=best&limit=5"
```

### Step 5 — Fetch the ORIGINAL document by id (what the UI will show)

```bash
curl -s "localhost:8007/doc?dataset=beir/quora/test&doc_id=1"
# -> {"doc_id":"1","text":"What is the step by step guide to invest in share market in india?"}
```

That text is the **original** document (never the preprocessed form) — read **by id** from
MongoDB. This is the path the UI uses to display the top‑k results.

### (Optional) Delete a dataset — independently, without touching the other

```bash
curl -s -X DELETE "localhost:8000/datasets?dataset=beir/quora/test&files=true&docs=true"
# -> {"files_deleted":true,"docs_deleted":500}   # deleting one never affects the others
```

---

## 4.1 Full dataset vs. a quick smoke run

Every `limit`/range above is just for **fast smoke tests** — **omit them to process the whole dataset**.

| Goal | Call |
|------|------|
| **Download once (internet)** | `POST /datasets/download {"dataset":"beir/quora/test"}` → fetch all ~522,931 Quora docs |
| **Full ingest to Mongo** (offline) | `POST /datasets/ingest {"dataset":"beir/quora/test"}` |
| **Verify it's complete** | `GET :8007/dataset/status?dataset=beir/quora/test&with_total=true` → `{"downloaded":true,"ingested_count":522931,"doc_count":522931,"fully_ingested":true}` |
| **Full inverted index** (offline) | `POST /datasets/index {"dataset":"beir/quora/test"}` → preprocesses the **entire** corpus (**slow** — tens of minutes) |
| **Index a slice only** | `POST /datasets/index {"dataset":"beir/quora/test","start":0,"stop":50000}` |
| Quick smoke | add `"limit":500` to ingest/index |

How the **download** works: it fetches the corpus **once** into `data/dataset/<id>/` (under
`IR_DATASETS_HOME`) and writes a `.manifest.json`. It's the **only** action that hits the
internet — ingest and build run offline and **refuse** (409) until it's done. After a full run
everything lives under `data/`: corpus in `data/dataset/`, raw docs in `data/mongo/`, index in
`data/artifacts/`.

---

## 5. Service reference

### 5.1 `preprocessing-service` (:8001) — normalize text

The **same** normalization is applied to documents (at index time) and to queries (at search
time) so they stay comparable. Pipeline: fold accents → lowercase → tokenize → drop stopwords
→ lemmatize (or stem).

```bash
curl -s -X POST localhost:8001/preprocess \
  -H "Content-Type: application/json" \
  -d '{"text":"The studies on investing in India!"}'
# -> {"tokens":["study","invest","india"],"normalized_text":"study invest india"}
```

- `POST /preprocess` — one text → `{tokens, normalized_text}`.
- `POST /preprocess/batch` — `{texts:[...]}` → one result per text (used to preprocess a corpus).
- Options (all optional, sensible defaults): `lowercase`, `remove_stopwords`, `stem`,
  `lemmatize`, `min_token_length`.

### 5.2 `doc-store-service` (:8007) — the dataset's facts in MongoDB

Owns the dataset's persisted ground truth in **three collections**, each keyed by `dataset`:
`documents` (the **raw/original** text, keyed by `doc_id`), `queries` (test queries, keyed by
`query_id`), and `qrels` (relevance judgments, one row per `(query_id, doc_id)` — the TREC shape).
All are populated offline and read **by id** at query time — the docs path is the graded
"original by id" requirement; queries/qrels are read by id the same way so the evaluation and
query‑refinement services don't re‑read ir‑datasets each call. Other services never touch Mongo
directly.

| Endpoint | What it does |
|----------|--------------|
| `GET /dataset/info?dataset=<id>` | **details before download** (network‑free): `{doc_count, num_queries, num_qrels, has_qrels, downloaded}` from ir‑datasets metadata — for a UI preview. Needs no database |
| `POST /dataset/download` | **The only endpoint that uses the internet.** Starts a background download into `data/dataset/<id>/`; returns a `JobStatus`. Idempotent (`skipped`); `force:true` re‑fetches; concurrent → **409** |
| `POST /dataset/prepare` | **offline** background ingest of **docs + queries + qrels** to Mongo — requires download first (else **409**). Docs drive progress; queries/qrels are a quick final phase. Idempotent **per collection** (`force:true` re‑ingests all three). Returns a `JobStatus` |
| `GET /dataset/status?dataset=beir/quora/test` | readiness → `{downloaded, ingested_count, queries_count, qrels_count, doc_count, active_job}` (cheap). Add `&with_total=true` for `fully_ingested` (verify a **full** ingest) |
| `GET /jobs/{id}` · `GET /jobs?type=&dataset=` | poll one job / list jobs (download + ingest) |
| `DELETE /dataset?dataset=beir/quora/test&files=&docs=` | independently delete the corpus folder and/or Mongo data; `docs=true` clears **all three** collections (others unaffected) |
| `GET /doc?dataset=…&doc_id=…` | one original doc by id (404 if absent) |
| `POST /docs` | batch fetch docs by id → `{docs:[…], missing:[…]}` (used by retrieval for top‑k) |
| `GET /docs/list?dataset=…&offset=&limit=` | **browse** stored docs by page (ordered by `seq`) → `{total, offset, limit, docs:[{seq, doc_id, text}]}`. Backs the UI viewer **and** the index build source |
| `GET /queries?dataset=…&offset=&limit=` | **browse** stored test queries by page → `{total, offset, limit, queries:[{seq, query_id, text}]}` |
| `GET /query?dataset=…&query_id=…` | one test query's text by id (404 if absent) |
| `GET /qrels?dataset=…&query_id=…` | the gold set for one query → `{query_id, judgments:[{query_id, doc_id, relevance}]}` |
| `GET /qrels/list?dataset=…&offset=&limit=` | **browse** all judgments by page (ordered by `seq`) |
| `GET /qrels/all?dataset=…` | the full qrels as `{query_id:{doc_id:relevance}}` — the shape `ranx`/`pytrec_eval` consume |

```bash
curl -s -X POST localhost:8007/docs \
  -H "Content-Type: application/json" \
  -d '{"dataset":"beir/quora/test","doc_ids":["1","39","999999"]}'
# -> {"docs":[{"doc_id":"1","text":"What is the step ..."}, ...],"missing":["999999"]}
```

> **Why async download / ingest?** They can take minutes on 200K+ docs. Returning a job (instead
> of blocking the HTTP call) lets a UI show a real progress bar and keeps the API responsive.

### 5.3 `indexing-service` (:8002) — inverted index

Builds and serves the inverted index: per‑term **postings** with term frequency, plus per‑doc
**lengths** and the corpus **average length** (`avgdl`) — exactly what BM25/TF‑IDF need. It
does **not** store raw text (that's the doc‑store).

| Endpoint | What it does |
|----------|--------------|
| `POST /build` | start a background build **from the doc‑store** (MongoDB). `{dataset, options?, start?, stop?, limit?, force?}`. Whole stored corpus or a `seq` **range** `[start:stop)`. **Requires the dataset ingested** (else **409**); doc‑store/preprocessing down → **503**. Returns a `JobStatus`; concurrent → **409** |
| `GET /index/status?dataset=beir/quora/test` | `{built, stats, active_job}` — what's built + live "docs indexed" during a build |
| `GET /jobs/{id}` · `GET /jobs?dataset=` | poll one build / list build jobs |
| `GET /stats?dataset=beir/quora/test` | index summary (num_docs, vocab_size, avg_doc_length, …) |
| `GET /postings?dataset=beir/quora/test&term=…&limit=…` | postings for a **normalized** term → `{df, cf, postings:[{doc_id, tf}]}` |
| `GET /term?dataset=beir/quora/test&term=…` | `{df, cf}` for a term |
| `GET /doc?dataset=beir/quora/test&doc_id=…` | per‑doc info **in the index** (token `length`) — *not* the text |
| `GET /built` | dataset ids that have a persisted index |

> Terms in `/postings` and `/term` must be **already normalized** (lowercased + lemmatized),
> because the index is built from preprocessed tokens. The retrieval service will normalize
> queries automatically later. **Range ordering** is the dataset's native, stable order (by
> position) — not a date sort (most datasets, Quora included, have no date field).

### 5.4 `api-gateway` (:8000) — single entry point

| Endpoint | What it does |
|----------|--------------|
| `GET /catalog` | the dataset **options** + live status (download/ingest/index) for **only** the `DATASETS` in `.env` (never the internet) — what a UI renders |
| `GET /datasets/info?dataset=<id>` | dataset details (size, qrels) for preview **before download** — network‑free |
| `GET /datasets/docs?dataset=<id>&offset=&limit=` | **browse** the stored corpus by page (UI database viewer) |
| `POST /datasets/download` · `/ingest` · `/index` | thin proxies that start the offline jobs (body carries `{dataset, …}`) and return a `JobStatus` |
| `DELETE /datasets?dataset=&files=&docs=` | proxy to the doc‑store delete |
| `GET /jobs/{service}/{job_id}` | poll a job; `service` = `docstore` or `indexing` |

Routing the query‑time services (retrieval, evaluation, …) through the gateway is added in later
phases.

---

## 6. Data & persistence

Everything long‑term lives under [`../data/`](../data) (bind‑mounted into the containers), so the
whole project folder is portable — copy it to move or back up.

| Path | Holds | Mounted as |
|------|-------|-----------|
| `data/dataset/<id>/` | each downloaded dataset in its **own** folder (corpus + qrels + `.manifest.json`) | `./data:/app/data` |
| `data/artifacts/` | built inverted indexes | `./data:/app/data` |
| `data/mongo/` | MongoDB engine files (raw docs + queries + qrels, read by id) | `./data/mongo:/data/db` |

See [`../data/README.md`](../data/README.md) for backup/reset. To wipe and start over:

```bash
docker compose down            # stop (data on disk is kept)
rm -rf data/mongo              # drop the database  (re-create with /ingest)
rm -rf data/artifacts          # drop indexes       (rebuild with /index)
rm -rf data/dataset            # force a fresh dataset download
```

To drop **one** dataset only, use `DELETE /datasets?dataset=<id>` (or delete just its
`data/dataset/<id>/` folder) — the others are untouched.

---

## 7. Running one service alone & troubleshooting

- **Run standalone:** `docker compose up doc-store-service` (each service starts on its own;
  it degrades gracefully when a dependency is missing).
- **Logs:** `docker compose logs -f indexing-service`.
- **A `localhost:<port>` call hangs but the service is healthy?** An IPv6/`localhost` quirk in
  Docker Desktop's port proxy. Use `127.0.0.1:<port>` instead, or restart Docker Desktop once.
  Container‑to‑container traffic is unaffected, so jobs keep running.
- **Common responses:**
  - A **job** with `state:"failed"` → see its `error` field (e.g. preprocessing down mid‑build).
  - `503` from `/build` → the doc‑store or preprocessing‑service is down (a build needs both).
  - `503` from doc‑store → MongoDB isn't reachable yet.
  - `409` from an action → the same dataset already has that job in flight, **or** a build was
    requested before the dataset is **ingested** (run download → ingest → build in order).
  - `400` → unknown dataset (not in the `DATASETS` catalog); `404` → doc/term/job not present.
