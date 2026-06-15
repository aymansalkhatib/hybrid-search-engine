# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# Install deps first for better layer caching. A BuildKit cache mount keeps already
# downloaded wheels across build attempts, so a flaky/slow network resumes instead of
# re-fetching. CPU-only torch comes from the PyTorch index in its own step (so
# sentence-transformers resolves against it without pulling the much larger CUDA build).
COPY services/representation-service/requirements.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 300 --retries 12 \
        torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 300 --retries 12 -r requirements.txt

# Shared contracts + service code.
COPY shared/ ./shared/
COPY services/representation-service/app/ ./app/

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    # Cache the downloaded BERT weights on the mounted data volume (one-time download).
    HF_HOME=/app/data/models \
    SENTENCE_TRANSFORMERS_HOME=/app/data/models

EXPOSE 8003

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8003/health').status==200 else 1)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003"]
