# Geo Embedder — BAAI/bge-m3 для геологии
## Запуск внутри Docker-контейнера

---

## Структура проекта

```
geo_embedder/
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── .env.example
├── requirements.txt
├── scripts/
│   ├── 01_chunk_documents.py
│   ├── 02_generate_queries.py
│   ├── 03_mine_hard_negatives.py
│   └── 05_evaluate.py
├── training/
│   └── 04_train_embedder.py
├── serving/
│   └── 06_serve.py
├── raw_docs/    ← входные PDF/DOCX
├── data/        ← обработанные данные
├── models/      ← кэш HF + веса
└── output/      ← результаты
```

## Быстрый старт

```bash
mkdir -p raw_docs data models output
cp .env.example .env        # вставить OPENAI_API_KEY
cp your_docs/*.pdf raw_docs/

make build
docker compose up -d pipeline
make check-gpu              # проверить CUDA

make shell                  # войти в контейнер
# внутри контейнера запускать скрипты 01→02→03→04→05

make up-serve               # поднять /v1/embeddings endpoint
make test-serve             # → dim: 1024
```

## Заметки по образу

pytorch:2.1.0-cuda12.1-cudnn8-runtime — базовый образ.
Dockerfile добавляет: poppler-utils (PDF), libxml2-dev (DOCX), build-essential (rank-bm25), non-root user geouser, HF_HOME на примонтированный том.

cudnn8-runtime достаточен для fine-tuning через PyTorch.
-devel нужен только при компиляции CUDA-расширений из исходников.

## Open WebUI

Admin Panel → Settings → RAG:
  Embedding Model Engine  : OpenAI
  Embedding Base URL      : http://<server_ip>:8080
  Embedding Model         : geo-embedder
  API Key                 : dummy
