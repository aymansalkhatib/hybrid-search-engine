# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better layer caching. Evaluation owns no model artifacts
# (retrieval/representation do the scoring), so the only heavy dep is ranx (metrics).
COPY services/evaluation-service/requirements.txt ./requirements.txt
# ranx pulls the large numba/llvmlite wheels; a longer timeout + retries rides out a
# slow link instead of failing the whole build on one timed-out download.
RUN pip install --no-cache-dir --timeout 180 --retries 6 -r requirements.txt

# Shared contracts + service code.
COPY shared/ ./shared/
COPY services/evaluation-service/app/ ./app/

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

EXPOSE 8006

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8006/health').status==200 else 1)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8006"]
