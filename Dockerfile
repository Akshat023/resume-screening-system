FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl --no-deps

COPY . .

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" \
    || echo "BERT preload failed — will download at runtime"

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# HuggingFace Spaces requires port 7860
EXPOSE 7860

CMD ["gunicorn", "app:app", \
     "--workers", "1", \
     "--threads", "4", \
     "--timeout", "300", \
     "--bind", "0.0.0.0:7860", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]