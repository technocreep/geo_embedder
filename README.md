# Geo Embedder — BAAI/bge-m3 для геологии

[🇬🇧 English version](README_eng.md)

Пайплайн дообучения мультиязычной модели эмбеддингов [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) на геологических документах. Цель — получить доменно-специфическую модель с точным retrieval для RAG-систем (Open WebUI + ChromaDB).

Методология основана на **X-Intelligence 3.0** (TCL Research, 2025).

---

## Архитектура системы

| Компонент | Инструмент | Назначение |
|-----------|-----------|-----------|
| **Базовая модель** | BAAI/bge-m3 | Мультиязычная (ru+en), dense vectors 1024-dim |
| **Фреймворк обучения** | SentenceTransformers ≥ 2.7 | Fine-tuning, evaluation |
| **Сервинг** | FastAPI + uvicorn (`06_serve.py`) | OpenAI-compatible `/v1/embeddings` |
| **Подключение** | Open WebUI → RAG Settings → OpenAI Engine | HTTP API без ollama |
| **Векторное хранилище** | ChromaDB (встроен в Open WebUI) | Автоматически после настройки |

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
│   ├── 00_split_data.py           ← разделение train/test по документам
│   ├── 01_chunk_documents.py      ← чанкование PDF/DOCX, LLM-разметка доменов
│   ├── 02_generate_queries.py     ← генерация вопросов к чанкам (async LLM)
│   ├── 03_mine_hard_negatives.py  ← построение триплетов (BM25 + cross-domain + adversarial)
│   ├── 05_evaluate.py             ← оценка NDCG, Recall, MRR, ARR
│   └── estimate_price.py          ← предварительная оценка стоимости LLM-вызовов
├── training/
│   └── 04_train_embedder.py       ← дообучение: MNRL + CoSENT + periodic mining
├── serving/
│   └── 06_serve.py                ← FastAPI сервер эмбеддера
├── utils/
│   └── llm_price.json             ← цены моделей (за 1M токенов)
├── raw_docs/    ← входные PDF/DOCX
├── data/        ← обработанные данные
├── models/      ← кэш HF + веса
└── output/      ← результаты оценки
```

---

## Пайплайн

Скрипты выполняются в пронумерованном порядке:

| Шаг | Скрипт | Вход → Выход | LLM |
|-----|--------|--------------|-----|
| 1 | `01_chunk_documents.py` | `raw_docs/` → `all_chunks.jsonl` | `DOMAIN_DETECTION_MODEL` (классификация домена) |
| 2 | `02_generate_queries.py` | `all_chunks.jsonl` → `all_queries.jsonl` | `QUERY_MODEL` (3 вопроса на чанк) |
| 3 | `00_split_data.py` | chunks + queries → train/test splits | — |
| 4 | `03_mine_hard_negatives.py` | train splits → `training_triplets.jsonl` | `ADVERSARIAL_MODEL` (30% train-чанков) |
| 5 | `04_train_embedder.py` | triplets → `/models/finetuned-bge-m3-geo/` | — |
| 6 | `05_evaluate.py` | test splits → `eval_results.json` | — |
| 7 | `06_serve.py` | модель → FastAPI на порту 8080 | — |

### Разделение train/test (шаг 3)

Стратегия выбирается автоматически:
- **≥ 5 источников**: сплит по документам — тестовые чанки из документов, которых модель не видела при обучении (нет data leakage)
- **< 5 источников**: сплит по чанкам (fallback)

Пропорция по умолчанию: 80% train / 20% test.

---

## Подготовка данных

### Чанкование (шаг 1)

| Параметр | Значение | Обоснование |
|----------|----------|-------------|
| `chunk_size` | 512 символов | Оптимально для bge-m3 |
| `overlap` | 64 символа | Сохраняет контекст на границах |
| Мин. длина чанка | 50 символов | Фильтрация заголовков и колонтитулов |

Каждый чанк размечается одной из 10 геологических подобластей через LLM (`DOMAIN_DETECTION_MODEL`). Вызовы асинхронные с параллелизмом `--concurrency` (по умолчанию 10):

> стратиграфия · петрография · тектоника · геохимия · гидрогеология ·
> рудная_геология · сейсмика · гис_картография · геофизика · общая_геология

### Генерация вопросов (шаг 2)

Для каждого чанка `QUERY_MODEL` генерирует 3 вопроса, варьируя стиль от академического до производственного. Фильтрация удаляет тривиальные и невалидируемые вопросы.

### Hard Negative Mining (шаг 4)

Три стратегии в комбинации (X-Intelligence 3.0 §3.1):

| Стратегия | Метод | Пример |
|-----------|-------|--------|
| **BM25** | Лексически похожие чанки из другого документа | «Пористость Тюменской свиты» vs «Пористость Васюганской свиты» |
| **Cross-domain** | Семантически близкие чанки из другой подобласти (dense retrieval) | Запрос про коллектор → negative из рудной геологии |
| **Adversarial** | LLM-перефразировка с намеренным изменением ключевых фактов | «глубина 2450 м» → «глубина 3800 м» |

Adversarial генерация применяется к 30% train-чанков (`--adversarial_ratio`).

**Балансировка**: не более 20% триплетов от одной подобласти (`balance_by_subdomain`).

### Целевые объёмы датасета

| Сплит | Минимум | Цель |
|-------|---------|------|
| Train (триплеты) | 5 000 | 30 000–50 000 |
| Test (запросы) | 200 | 500 |

---

## Обучение (шаг 5)

### Функции потерь

Основное обучение использует два лосса одновременно:

| Loss | Назначение |
|------|-----------|
| **MultipleNegativesRankingLoss** | In-batch negatives; масштабируется с batch_size; аналог Contrast Loss из X-Intelligence §3.1 |
| **CoSENTLoss** | Scored пары (positive=1.0, negative=0.0); улучшает тонкую разметку |

**Periodic Hard Negative Mining** (`--periodic_mining`): запускается после основного обучения. Текущая модель переиндексирует корпус и добирает наиболее трудные примеры, на которых затем дообучается через **TripletLoss**.

**Dev-сет**: если файл dev не передан явно (`--dev`), скрипт автоматически выделяет 10% случайных триплетов в dev-сплит для мониторинга NDCG@10 в ходе обучения.

### Гиперпараметры по умолчанию

| Параметр | Значение |
|----------|----------|
| Base model | BAAI/bge-m3 |
| Batch size | 32 |
| Learning rate | 2e-5, cosine decay |
| Warmup | 10% от total steps |
| Epochs | 3 |
| Mixed precision | fp16 (только CUDA) |
| Eval / save | каждые ¼ эпохи |
| Best checkpoint | по `NDCG@10` на dev |

### MLflow

Если задана переменная `MLFLOW_TRACKING_URI`, все параметры и метрики логируются автоматически. При отсутствии — логирование молча отключается.

---

## Оценка (шаг 6)

Оценка на **test-сете** — чанки из документов, которые модель никогда не видела при обучении.

| Метрика | Целевое значение | Что измеряет |
|---------|-----------------|--------------|
| **NDCG@10** | > 0.72 | Позиционное качество ранжирования (лог-дисконт) |
| **Recall@5** | > 0.75 | Доля запросов с positive в топ-5 |
| **Recall@10** | > 0.85 | Доля запросов с positive в топ-10 |
| **MRR** | > 0.68 | Среднее 1/rank первого релевантного результата |
| **ARR@5** | > 0.80 | Доля запросов с хотя бы 1 релевантным в топ-5 |

Основная метрика для отбора чекпоинтов — **NDCG@10**.

### Прогресс по экспериментам (E0–E4)

| Эксп. | Изменение | Ожидаемый NDCG@10 |
|-------|-----------|------------------|
| E0 — Baseline | BAAI/bge-m3 без дообучения | ~0.52 |
| E1 — +BM25 negatives | Fine-tuning на BM25 триплетах | ~0.58 (+0.06) |
| E2 — +Cross-domain | Добавляем cross-domain negatives | ~0.63 (+0.05) |
| E3 — +Adversarial | Добавляем adversarial negatives | ~0.67 (+0.04) |
| E4 — +Periodic Mining | `--periodic_mining` | ~0.73 (+0.06) |

---

## Конфигурация (.env)

```bash
OPENAI_API_KEY=sk-...
OPENAI_API_BASE=https://openrouter.ai/api/v1

DOMAIN_DETECTION_MODEL=nvidia/nemotron-3-nano-30b-a3b  # шаг 1
QUERY_MODEL=anthropic/claude-haiku-4.5                 # шаг 2
ADVERSARIAL_MODEL=openai/gpt-4o-mini                   # шаг 4

MODEL_PATH=/models/finetuned-bge-m3-geo

MLFLOW_TRACKING_URI=http://172.17.0.1:5000
MLFLOW_EXPERIMENT_NAME=geo-embedder

HF_TOKEN=hf_...
# HTTPS_PROXY=socks5://172.17.0.1:1080
```

---

## Быстрый старт

```bash
mkdir -p raw_docs data models output
cp .env.example .env       # вставить ключи
cp your_docs/*.pdf raw_docs/

make build
make up                    # поднять контейнер pipeline

make check-gpu             # проверить CUDA
make shell                 # войти в контейнер

# Предварительная оценка стоимости LLM-вызовов:
python scripts/estimate_price.py --num_chunks 1000

# Полный пайплайн одной командой:
make run-pipeline
```

### Ручной запуск отдельных шагов (внутри контейнера)

```bash
python scripts/01_chunk_documents.py \
    --input_dir /data/raw_docs --output /data/processed/all_chunks.jsonl

python scripts/02_generate_queries.py \
    --chunks /data/processed/all_chunks.jsonl \
    --output /data/processed/all_queries.jsonl

python scripts/00_split_data.py \
    --chunks /data/processed/all_chunks.jsonl \
    --queries /data/processed/all_queries.jsonl \
    --output_dir /data/processed --test_ratio 0.2

python scripts/03_mine_hard_negatives.py \
    --queries /data/processed/train_queries.jsonl \
    --chunks /data/processed/train_chunks.jsonl \
    --output /data/processed/training_triplets.jsonl

python training/04_train_embedder.py \
    --triplets /data/processed/training_triplets.jsonl \
    --output_dir /models/finetuned-bge-m3-geo --epochs 3

python scripts/05_evaluate.py \
    --test /data/processed/test_queries.jsonl \
    --chunks /data/processed/test_chunks.jsonl \
    --models "BAAI/bge-m3" "/models/finetuned-bge-m3-geo" \
    --labels "Baseline" "Fine-tuned" \
    --output /output/eval_results.json
```

---

## Команды Makefile

```bash
make build           # сборка Docker-образа
make up              # запуск контейнера pipeline
make down            # остановка
make shell           # войти в контейнер
make run-pipeline    # полный пайплайн (01→02→00→03→04→05)
make eval-baseline   # оценка baseline BAAI/bge-m3
make eval-finetuned  # оценка дообученной модели
make check-gpu       # проверить CUDA внутри контейнера
```

> **Примечание**: сервис `serve` в `docker-compose.yml` закомментирован.
> Для запуска serving раскомментируйте секцию `serve` и используйте `make up-serve`.

---

## Заметки по Docker-образу

Базовый образ: `pytorch:2.1.0-cuda12.1-cudnn8-runtime`

Dockerfile добавляет: `poppler-utils` (PDF), `libxml2-dev` (DOCX), `build-essential` (rank-bm25), non-root user `geouser`, `HF_HOME` на примонтированный том.

`cudnn8-runtime` достаточен для fine-tuning через PyTorch. `-devel` нужен только при компиляции CUDA-расширений из исходников.

Тома при запуске:

| Хост | Контейнер | Режим |
|------|-----------|-------|
| `./raw_docs` | `/data/raw_docs` | ro |
| `./data` | `/data/processed` | rw |
| `./models` | `/models` | rw |
| `./output` | `/output` | rw |
| `./scripts` | `/app/scripts` | rw (hot-reload) |
| `./training` | `/app/training` | rw (hot-reload) |

> `utils/` не монтируется — запекается в образ при `make build`.

---

## Подключение к Open WebUI

```
Admin Panel → Settings → RAG:
  Embedding Model Engine  : OpenAI
  Embedding Base URL      : http://<server_ip>:8080
  Embedding Model         : geo-embedder
  API Key                 : dummy
```

Эндпоинты сервера: `POST /v1/embeddings`, `GET /v1/models`, `GET /health`.
