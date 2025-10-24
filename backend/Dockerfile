# backend/Dockerfile — FastAPI API
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m appuser
WORKDIR /app

COPY backend/requirements.txt ./backend/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r backend/requirements.txt

COPY backend ./backend
COPY shared ./shared

ENV PYTHONPATH="/app:${PYTHONPATH}"

EXPOSE 8000
USER appuser
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
