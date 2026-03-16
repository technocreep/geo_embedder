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

## Метрики оценки

Скрипт `scripts/05_evaluate.py` вычисляет следующие метрики на тестовой выборке (corpus = test_chunks, queries = test_queries).

### NDCG@k — Normalized Discounted Cumulative Gain

Оценивает **позиционное качество ранжирования**: правильный чанк должен быть как можно выше в топ-k результатах. Позиции далеко от первой штрафуются логарифмическим дисконтированием. Значение от 0 до 1; 1.0 — правильный чанк всегда на первом месте.

> Järvelin, K., Kekäläinen, J. *Cumulated gain-based evaluation of IR techniques*. ACM TOIS, 2002.
> https://dl.acm.org/doi/10.1145/582415.582418

### Recall@k

Доля запросов, для которых **правильный чанк попал в топ-k**. Не учитывает позицию внутри топ-k. Recall@5 < Recall@10 всегда. Значение от 0 до 1.

> Manning, C. D., Raghavan, P., Schütze, H. *Introduction to Information Retrieval*. Cambridge UP, 2008. §8.4
> https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html

### MRR — Mean Reciprocal Rank

Среднее значение `1 / rank` первого релевантного результата по всем запросам. Фокусируется на том, **насколько быстро** пользователь найдёт правильный ответ. Если правильный чанк на 1-м месте — вклад 1.0; на 2-м — 0.5; на 3-м — 0.333 и т.д.

> Voorhees, E. M. *The TREC-8 Question Answering Track Report*. TREC 1999.
> https://trec.nist.gov/pubs/trec8/papers/qa_report.pdf

### MAP@100 — Mean Average Precision

Среднее значение Average Precision (площадь под кривой precision–recall) по всем запросам в топ-100. Чувствительна к порядку и полноте; полезна при множественных релевантных документах. При одном правильном ответе MAP@100 ≈ MRR.

> Buckley, C., Voorhees, E. M. *Retrieval evaluation with incomplete information*. SIGIR 2004.
> https://dl.acm.org/doi/10.1145/1008992.1009000

### Сводная таблица

| Метрика     | Учитывает позицию | Требует всех правильных | Диапазон |
|-------------|:-----------------:|:-----------------------:|:--------:|
| NDCG@k      | Да (лог-дисконт)  | Нет                     | [0, 1]   |
| Recall@k    | Нет               | Нет                     | [0, 1]   |
| MRR         | Да (1/rank)       | Нет                     | (0, 1]   |
| MAP@100     | Да                | Нет                     | [0, 1]   |

Основная метрика для отбора чекпоинтов — **NDCG@10**.

---

## Open WebUI

Admin Panel → Settings → RAG:
  Embedding Model Engine  : OpenAI
  Embedding Base URL      : http://<server_ip>:8080
  Embedding Model         : geo-embedder
  API Key                 : dummy
