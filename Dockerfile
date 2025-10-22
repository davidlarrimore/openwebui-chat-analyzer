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

RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app
RUN chown appuser:appuser /app
COPY --chown=appuser:appuser backend/ /app/backend

USER appuser

EXPOSE 8502
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8502"]

# ------------------------------------------------------------
# Frontend Next.js image
# ------------------------------------------------------------
FROM node:20-alpine AS frontend-next-builder

WORKDIR /app
ENV NEXT_TELEMETRY_DISABLED=1 \
    NODE_ENV=production

RUN apk add --no-cache libc6-compat && corepack enable

COPY frontend-next/pnpm-lock.yaml frontend-next/package.json ./
RUN pnpm install --frozen-lockfile

COPY frontend-next/ .
RUN pnpm build

FROM node:20-alpine AS frontend-next

WORKDIR /app
ENV NEXT_TELEMETRY_DISABLED=1 \
    NODE_ENV=production

RUN apk add --no-cache libc6-compat && corepack enable

COPY --from=frontend-next-builder /app/public ./public
COPY --from=frontend-next-builder /app/.next ./.next
COPY --from=frontend-next-builder /app/package.json ./package.json
COPY --from=frontend-next-builder /app/pnpm-lock.yaml ./pnpm-lock.yaml
COPY --from=frontend-next-builder /app/node_modules ./node_modules
COPY --from=frontend-next-builder /app/next.config.mjs ./next.config.mjs

EXPOSE 3000
CMD ["pnpm", "start"]
