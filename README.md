# IR Project 2026 — Information Retrieval System

A professional **search engine** built over **two large IR datasets** (each > 200K documents),
using classic and modern retrieval techniques, and designed as a clean
**Service-Oriented Architecture (SOA)**: every capability is an independent **Docker** service,
orchestrated together with **Docker Compose**.

> **Status:** 🚧 Early scaffolding. We build step-by-step.
> This README will grow as services land.

---

## Documents


---

## What it does (target)

- Retrieve relevant documents for natural-language queries over **two datasets**.
- Four representations: **TF-IDF (VSM)**, **Embeddings (Word2Vec/BERT)**, **BM25**, and
  **Hybrid** (Serial + Parallel with score fusion).
- **Inverted index**, query preprocessing & refinement, ranking, and standard
  **evaluation** (MAP, Recall, P@10, nDCG).
- A **Streamlit UI** to pick the dataset, choose the model, tune BM25 params, and switch
  between Serial/Parallel hybrid.

## Datasets (default, swappable via `.env`)

| Slot | ir-datasets ID | ~Docs | Nature |
|------|----------------|-------|--------|
| A | `beir/quora/test` | ~522K | Short question texts |
| B | `wikir/en1k/test` | ~370K | Wikipedia articles |

The data layer is **dataset-agnostic**: changing a dataset is a single value in `.env`.

## Architecture

Microservices (FastAPI) behind an **API Gateway**, with a **Streamlit** UI.

| Service | Port | Responsibility |
|---------|------|----------------|
| api-gateway | 8000 | Single entry point / orchestration |
| preprocessing-service | 8001 | Normalize, tokenize, stem, lemmatize |
| indexing-service | 8002 | Inverted index |
| representation-service | 8003 | TF-IDF / embeddings / BM25 |
| retrieval-service | 8004 | Matching, ranking, hybrid + fusion |
| query-refinement-service | 8005 | Correction, expansion, suggestion |
| evaluation-service | 8006 | MAP / Recall / P@10 / nDCG |
| ui | 8501 | Streamlit web UI |

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

## License

Educational project — University IR course, 2026.
