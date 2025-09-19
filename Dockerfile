# Dockerfile at REPO ROOT
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install backend deps
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source folders needed by FastAPI
COPY backend/ ./backend
COPY shared/  ./shared

# Ensure Python can import from /app (so "from shared..." works)
ENV PYTHONPATH=/app

# Cloud Run will set $PORT (we default to 8080)
ENV PORT=8080

# Start FastAPI explicitly by module path (robust to workdir)
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8080"]
