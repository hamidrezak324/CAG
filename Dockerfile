FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
COPY wheels/ ./wheels/
RUN pip install --no-cache-dir --find-links=./wheels -r requirements.txt

COPY . .
COPY dataset ./dataset
COPY chroma_db ./chroma_db
COPY .cache/huggingface /app/.cache/huggingface

ENV HF_HOME=/app/.cache/huggingface

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]