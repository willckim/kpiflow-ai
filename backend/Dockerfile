# backend/Dockerfile (build from repo ROOT with: -f backend/Dockerfile .)
FROM python:3.11-slim

WORKDIR /app

# Keep image light and logs unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ---- Install deps ----
# Copy backend requirements into a temp path in the image
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ---- Copy source ----
# Copy ONLY the code we need from the repo root into /app
COPY backend/ ./        # puts backend files (including app.py) into /app
COPY shared/ ./shared   # include your shared module for imports

# Cloud Run provides $PORT; default to 8080 if not set
ENV PORT=8080

# Start FastAPI via Uvicorn (app.py defines: app = FastAPI(...))
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
