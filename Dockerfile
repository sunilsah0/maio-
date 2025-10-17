# syntax=docker/dockerfile:1

FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1
WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
	curl \
	&& rm -rf /var/lib/apt/lists/*

# Install deps first for caching
COPY pyproject.toml README.md /app/
RUN pip install --no-cache-dir --upgrade pip && \
	pip install --no-cache-dir .

# Copy application
COPY src/ /app/src/
# Copy model artifacts if present at build time (for releases)
COPY model/ /app/model/

ENV PORT=8080 \
	MODEL_DIR=/app/model \
	PYTHONPATH=/app/src

EXPOSE 8080

CMD ["uvicorn", "maio.app:app", "--host", "0.0.0.0", "--port", "8080"]

