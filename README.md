# IR Project 2026 — Information Retrieval System

A professional **search engine** built over a **large IR dataset** (> 200K documents; a second
dataset is an optional bonus), using classic and modern retrieval techniques, and designed as a clean
**Service-Oriented Architecture (SOA)**: every capability is an independent **Docker** service,
orchestrated together with **Docker Compose**.

> **Status:** 🚧 Early scaffolding. We build step-by-step.
> This README will grow as services land.

---

## Documents

- **Usage guide (how to run & use each service):** [`docs/USAGE.md`](docs/USAGE.md)

---

## What it does (target)

- Retrieve relevant documents for natural-language queries over the dataset (a second is bonus).
- Four representations: **TF-IDF (VSM)**, **Embeddings (Word2Vec/BERT)**, **BM25**, and
  **Hybrid** (Serial + Parallel with score fusion).
- **Inverted index**, query preprocessing & refinement, ranking, and standard
  **evaluation** (MAP, Recall, P@10, nDCG).
- A **Streamlit UI** to pick the dataset, choose the model, tune BM25 params, and switch
  between Serial/Parallel hybrid.

## Datasets (default, swappable via `.env`)

Only **one** dataset is required; a **second is a bonus**. The dataset **must** have `qrels`.

| Slot | ir-datasets ID | ~Docs | Nature |
|------|----------------|-------|--------|
| A *(required)* | `beir/quora/test` | ~522K | Short question texts |
| B *(bonus)* | `wikir/en1k/test` | ~370K | Wikipedia articles |

The data layer is **dataset-agnostic**: changing a dataset is a single value in `.env`.

## Architecture

Microservices (FastAPI) behind an **API Gateway**, with a **Streamlit** UI.

**The API Gateway is the single external door (§4):** every functional service is
**internal-only** (compose network) and reached **through the gateway at `:8000`** under a
per-service prefix. Only three containers publish a host port — the gateway, the dev/QA
**Verification Console** (`:8090`), and **mongo-express** (`:8081`).

| Service | Port | Reached via | Responsibility |
|---------|------|-------------|----------------|
| api-gateway | 8000 | **published** | Single external entry point / orchestration |
| preprocessing-service | 8001 | `:8000/preprocessing` | Normalize, tokenize, stem, lemmatize |
| indexing-service | 8002 | `:8000/indexing` | Inverted index |
| representation-service | 8003 | `:8000/representation` | TF-IDF / embeddings / BM25 / hybrid + search |
| retrieval-service | 8004 | _(planned)_ | Matching, ranking, hybrid + fusion |
| query-refinement-service | 8005 | _(planned)_ | Correction, expansion, suggestion |
| evaluation-service | 8006 | _(planned)_ | MAP / Recall / P@10 / nDCG |
| doc-store-service | 8007 | `:8000/docstore` + `/catalog`, `/datasets/*` | Docs + queries + qrels in MongoDB, read by ID |
| dashboard-service | 8090 | **published** | Dev/QA Verification Console (proxies every service) |
| ui | 8501 | _(planned)_ | Streamlit web UI |

## Tech stack

Python · FastAPI · Streamlit · ir-datasets · scikit-learn · bm25s · sentence-transformers ·
gensim · FAISS / Qdrant · ranx · Docker / Docker Compose.

## Getting started

> Requires **Docker Desktop**.

```bash
# (once services exist)
cp .env.example .env       # configure datasets, ports, models
docker compose up          # run the whole system
# or run a single service:
docker compose up api-gateway
```

## Data & persistence

Everything long-term — the **dataset cache**, **built indexes**, and the **MongoDB database** —
lives in [`data/`](data/) (`raw/`, `artifacts/`, `mongo/`), **bind-mounted** into the
containers. So all data sits in the **project root**: portable (copy the folder to move or back
up) and persistent across restarts/rebuilds, instead of hidden in Docker's internal storage.
See [`data/README.md`](data/README.md) for layout, backup, and reset. Data contents are gitignored.

## License

Educational project — University IR course, 2026.
