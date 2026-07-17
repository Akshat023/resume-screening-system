# ── Stage 1: Base ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# System dependencies needed for pdfminer, psycopg2, spaCy
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Stage 2: Python dependencies ──────────────────────────────────────────────
COPY requirements.txt .

# Install CPU-only torch first (avoids pulling GPU version as a dep)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install all other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model via direct URL (avoids 404 bug with python -m spacy download)
RUN pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl --no-deps

# ── Stage 3: App code ─────────────────────────────────────────────────────────
COPY . .

# Pre-download BERT model so container startup is fast
# This bakes the model into the image (~90MB) rather than downloading at runtime
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" \
    || echo "BERT preload failed — will download at runtime"

# ── Stage 4: Runtime ──────────────────────────────────────────────────────────
# Non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 5000

# Gunicorn: 1 worker, 4 threads, 5min timeout for BERT scoring
CMD ["gunicorn", "app:app", \
     "--workers", "1", \
     "--threads", "4", \
     "--timeout", "300", \
     "--bind", "0.0.0.0:5000", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]