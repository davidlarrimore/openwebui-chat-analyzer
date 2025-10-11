# syntax=docker/dockerfile:1

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    NLTK_DATA="/opt/nltk_data"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv

# ------------------------------------------------------------
# Backend image
# ------------------------------------------------------------
FROM base AS backend-builder

WORKDIR /tmp/backend
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

FROM base AS backend

COPY --from=backend-builder /opt/venv /opt/venv
WORKDIR /app
COPY backend/ /app/backend

EXPOSE 8502
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8502"]

# ------------------------------------------------------------
# Frontend image
# ------------------------------------------------------------
FROM base AS frontend-builder

WORKDIR /tmp/frontend
COPY frontend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
RUN mkdir -p /opt/nltk_data && python -m textblob.download_corpora

FROM base AS frontend

COPY --from=frontend-builder /opt/venv /opt/venv
COPY --from=frontend-builder /opt/nltk_data /opt/nltk_data

WORKDIR /app
COPY frontend/ /app/frontend

EXPOSE 8501
CMD ["streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
