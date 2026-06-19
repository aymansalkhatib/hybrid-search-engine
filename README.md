# IR Project 2026 — Information Retrieval System

A professional **search engine** built over a **large IR dataset** (> 200K documents; a second
dataset is an optional bonus), using classic and modern retrieval techniques, and designed as a clean
**Service-Oriented Architecture (SOA)**: every capability is an independent **Docker** service,
orchestrated together with **Docker Compose**.

> **Status:** ✅ Core system built. The full mandatory pipeline runs end-to-end — preprocessing,
> inverted index, the four representations (TF-IDF, BM25, Word2Vec, BERT) + **Hybrid** (Serial &
> Parallel with fusion), query refinement, evaluation (MAP/nDCG/Recall/P@10), and an interactive
> console UI. Three **extra features** are implemented: **Query Refinement**, **Document
> Clustering**, and **Topic Detection** — each independently toggleable and evaluable before/after.
> Remaining work is non-code: full evaluation runs + charts and the written report.

---

## Documents

- **Data layer (layout / backup / reset):** [`data/README.md`](data/README.md)

---

## What it does

- Retrieve relevant documents for natural-language queries over the dataset (a second is bonus).
- Four representations: **TF-IDF (VSM)**, **Embeddings (Word2Vec/BERT)**, **BM25**, and
  **Hybrid** (Serial re-rank + Parallel fusion).
- **Inverted index** (+ Boolean search), shared query/doc **preprocessing**, **query refinement**
  (spell-correct / expand), ranking, and standard **evaluation** (MAP, Recall, P@10, nDCG) with
  **before/after** comparison.
- **Extra features:** Query Refinement, Document Clustering (KMeans), Topic Detection (LDA) —
  each toggleable in the UI and evaluable on its own.
- An interactive **console UI** (the Verification Console) to pick the dataset, choose the model,
  tune BM25 `k1`/`b`, switch Serial/Parallel hybrid, toggle **Basic vs Basic + extras**, and view
  results & evaluation charts.

## Datasets (default, swappable via `.env`)

Only **one** dataset is required; a **second is a bonus**. The dataset **must** have `qrels`.

| Slot | ir-datasets ID | ~Docs | Nature |
|------|----------------|-------|--------|
| A *(required)* | `beir/quora/test` | ~522K | Short question texts |
| B *(bonus)* | `wikir/en1k/test` | ~370K | Wikipedia articles |

The data layer is **dataset-agnostic**: changing a dataset is a single value in `.env` (`DATASETS`).

## Architecture

Microservices (FastAPI) behind an **API Gateway**, with an interactive web **console** UI.

**The API Gateway is the single external door (§4):** every functional service is
**internal-only** (compose network) and reached **through the gateway at `:8000`** under a
per-service prefix. Only three containers publish a host port — the gateway, the dev/QA
**Verification Console** (`:8090`), and **mongo-express** (`:8081`).

| Service | Port | Reached via | Responsibility |
|---------|------|-------------|----------------|
| api-gateway | 8000 | **published** | Single external entry point / orchestration |
| preprocessing-service | 8001 | `:8000/preprocessing` | Normalize, tokenize, stopwords, stem, lemmatize |
| indexing-service | 8002 | `:8000/indexing` | Inverted index (df · tf · avgdl) + Boolean match |
| representation-service | 8003 | `:8000/representation` | TF-IDF / Word2Vec / BERT / BM25 + scoring primitives |
| retrieval-service | 8004 | `:8000/retrieval` | Match & rank · Hybrid + fusion · Boolean · cluster/topic re-rank |
| query-refinement-service | 8005 | `:8000/refinement` | Spell-correction · synonym expansion · suggestion |
| evaluation-service | 8006 | `:8000/evaluation` | MAP · nDCG · Recall · P@10 · before/after reports |
| doc-store-service | 8007 | `:8000/docstore` (+ `/catalog`, `/datasets/*`) | Docs + queries + qrels in MongoDB, read by ID |
| clustering-service | 8008 | `:8000/clustering` | Document clustering (TF-IDF + KMeans) — extra feature |
| topic-service | 8009 | `:8000/topic` | Topic modelling (LDA) — extra feature |
| doc-store-db (MongoDB) | 27017 | internal | The raw-doc / queries / qrels database |
| dashboard-service | 8090 | **published** | Verification Console — interactive UI / proxies every service |
| mongo-express | 8081 | **published** | MongoDB admin UI (dev/QA) |

## Tech stack

Python · FastAPI · ir-datasets · scikit-learn (TF-IDF, KMeans, LDA) · bm25s · sentence-transformers ·
gensim · symspellpy · NLTK (WordNet) · MongoDB (pymongo) · ranx · Docker / Docker Compose.

## Getting started

> Requires **Docker Desktop**.

```bash
cp .env.example .env       # configure datasets, ports, models
docker compose up          # run the whole system
# or run a single service alone (SOA — each runs independently):
docker compose up api-gateway
```

Then open the **Verification Console** at <http://localhost:8090> to drive the system
(prepare a dataset → build representations → search → evaluate), or call the API Gateway at
<http://localhost:8000/docs>.

## Data & persistence

Everything long-term — the **dataset cache**, **built artifacts** (index, representations,
clustering, topic models, evaluation reports), and the **MongoDB database** — lives in
[`data/`](data/) (`dataset/`, `artifacts/`, `mongo/`), **bind-mounted** into the containers. So all
data sits in the **project root**: portable (copy the folder to move or back up) and persistent
across restarts/rebuilds, instead of hidden in Docker's internal storage. See
[`data/README.md`](data/README.md) for layout, backup, and reset. Data contents are gitignored.

## License

Educational project — University IR course, 2026.
