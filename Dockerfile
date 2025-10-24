# Dockerfile (root) — Streamlit UI
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m appuser
WORKDIR /app

# Copy requirements first for caching
COPY frontend/requirements.txt ./frontend/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r frontend/requirements.txt

# Copy source code
COPY frontend ./frontend
COPY shared ./shared

ENV PYTHONPATH="/app:${PYTHONPATH}"

EXPOSE 10000
USER appuser

CMD ["streamlit", "run", "frontend/app.py", "--server.port=$PORT", "--server.address=0.0.0.0"]
