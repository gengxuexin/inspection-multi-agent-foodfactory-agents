FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FQA_DATA_DIR=/app/data

# 依赖（含 pillow 的 C 库）
RUN apt-get update \
 && apt-get install -y --no-install-recommends libjpeg62-tz curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY knowledge ./knowledge
COPY tests ./tests

RUN mkdir -p /app/data/uploads

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD curl -fsS http://127.0.0.1:8000/api/stats || exit 1

# 生产建议用 gunicorn 多 worker（注意 SQLite 写锁，生产换 PostgreSQL）
CMD ["uvicorn", "app.api.service:app", "--host", "0.0.0.0", "--port", "8000"]
