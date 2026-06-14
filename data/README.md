# `data/` — datasets, artifacts & database (contents gitignored)

This folder is the project's **persistent data layer**. Everything long‑term lives here and is
**bind‑mounted** into the containers, so it all sits in the **project root** — portable and
persistent across restarts/rebuilds.

## Layout

| Path | What | Created by |
|------|------|------------|
| `dataset/` | downloaded datasets, set via `IR_DATASETS_HOME`. Each dataset lives in its **own subfolder** (`dataset/beir/quora/test/…`) with a `.manifest.json` marking a complete download — so datasets are independently deletable | the `POST /dataset/download` endpoint |
| `artifacts/` | built artifacts: inverted indexes (later: TF‑IDF/embeddings/BM25) | the `/build` endpoints |
| `mongo/` | MongoDB engine files — the **raw document store** read by id at query time | the `doc-store-db` (MongoDB) container |

All three are created automatically at runtime (`./data:/app/data` for the cache + artifacts,
`./data/mongo:/data/db` for MongoDB).

## Why bind mounts (not Docker *named* volumes)?

Bind mounts keep **all** data inside the project, so it's easy to **move, back up, and inspect**
— copy this folder and everything (datasets, indexes, and the database) goes with it. A Docker
*named* volume would live in Docker's internal storage (the WSL2 VM on Windows), which isn't
portable.

> **Note on MongoDB:** running Mongo's engine files on a bind mount works on this setup
> (Docker Desktop + WSL2/virtiofs). If on another machine Mongo fails to start with a
> WiredTiger/locking error, switch `doc-store-db` back to a named volume in
> `docker-compose.yml` — the raw docs are always re‑ingestable via `POST /dataset/prepare`.

## Backup / move / reset

- **Back up or move everything:** `docker compose down`, then copy the whole `data/` folder.
- **Reset the database only:** `docker compose down`, delete `data/mongo/`, then re‑run prepare.
- **Reset indexes:** delete `data/artifacts/`. **Re‑download the dataset:** delete `data/dataset/`.

> Only this README is tracked in git; the (large) data, artifacts and database files are ignored.
