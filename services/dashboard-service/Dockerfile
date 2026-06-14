# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better layer caching.
COPY services/dashboard-service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Shared contracts + service code + the static front-end.
COPY shared/ ./shared/
COPY services/dashboard-service/app/ ./app/
COPY services/dashboard-service/frontend/ ./frontend/

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

EXPOSE 8090

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8090/healthz').status==200 else 1)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8090"]
