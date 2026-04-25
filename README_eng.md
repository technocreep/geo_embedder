# Geo Embedder — BAAI/bge-m3 for Geology

[🇷🇺 Русская версия](README.md)

A fine-tuning pipeline for the multilingual embedding model [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) on geological documents. The goal is to produce a domain-specific model with precise retrieval for RAG systems (Open WebUI + ChromaDB).

Methodology is based on **X-Intelligence 3.0** (TCL Research, 2025).

---

## System Architecture

| Component | Tool | Purpose |
|-----------|------|---------|
| **Base model** | BAAI/bge-m3 | Multilingual (ru+en), dense vectors 1024-dim |
| **Training framework** | SentenceTransformers ≥ 2.7 | Fine-tuning, evaluation |
| **Serving** | FastAPI + uvicorn (`06_serve.py`) | OpenAI-compatible `/v1/embeddings` |
| **Integration** | Open WebUI → RAG Settings → OpenAI Engine | HTTP API without ollama |
| **Vector store** | ChromaDB (built into Open WebUI) | Automatic after configuration |

---

## Project Structure

```
geo_embedder/
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── .env.example
├── requirements.txt
├── scripts/
│   ├── 00_split_data.py           ← train/test split by documents
│   ├── 01_chunk_documents.py      ← PDF/DOCX chunking, LLM domain tagging
│   ├── 02_generate_queries.py     ← query generation per chunk (async LLM)
│   ├── 03_mine_hard_negatives.py  ← triplet construction (BM25 + cross-domain + adversarial)
│   ├── 05_evaluate.py             ← NDCG, Recall, MRR, ARR evaluation
│   └── estimate_price.py          ← pre-run LLM cost estimation
├── training/
│   └── 04_train_embedder.py       ← fine-tuning: MNRL + CoSENT + periodic mining
├── serving/
│   └── 06_serve.py                ← FastAPI embedder server
├── utils/
│   └── llm_price.json             ← model prices (per 1M tokens)
├── raw_docs/    ← input PDF/DOCX files
├── data/        ← processed data
├── models/      ← HF cache + weights
└── output/      ← evaluation results
```

---

## Pipeline

Scripts run in numbered order:

| Step | Script | Input → Output | LLM |
|------|--------|----------------|-----|
| 1 | `01_chunk_documents.py` | `raw_docs/` → `all_chunks.jsonl` | `DOMAIN_DETECTION_MODEL` (domain classification) |
| 2 | `02_generate_queries.py` | `all_chunks.jsonl` → `all_queries.jsonl` | `QUERY_MODEL` (3 queries per chunk) |
| 3 | `00_split_data.py` | chunks + queries → train/test splits | — |
| 4 | `03_mine_hard_negatives.py` | train splits → `training_triplets.jsonl` | `ADVERSARIAL_MODEL` (30% of train chunks) |
| 5 | `04_train_embedder.py` | triplets → `/models/finetuned-bge-m3-geo/` | — |
| 6 | `05_evaluate.py` | test splits → `eval_results.json` | — |
| 7 | `06_serve.py` | model → FastAPI on port 8080 | — |

### Train/Test Split (step 3)

Strategy is selected automatically:
- **≥ 5 sources**: document-level split — test chunks come from documents the model never saw during training (no data leakage)
- **< 5 sources**: chunk-level split (fallback)

Default ratio: 80% train / 20% test.

---

## Data Preparation

### Chunking (step 1)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `chunk_size` | 512 characters | Optimal for bge-m3 |
| `overlap` | 64 characters | Preserves context at boundaries |
| Min chunk length | 50 characters | Filters out headers and footers |

Each chunk is tagged with one of 10 geological sub-domains via LLM (`DOMAIN_DETECTION_MODEL`). Calls are async with `--concurrency` parallelism (default: 10):

> stratigraphy · petrography · tectonics · geochemistry · hydrogeology ·
> ore_geology · seismics · gis_cartography · geophysics · general_geology

### Query Generation (step 2)

For each chunk, `QUERY_MODEL` generates 3 questions varying in style from academic to applied. Filtering removes trivial and unverifiable questions.

### Hard Negative Mining (step 4)

Three strategies combined (X-Intelligence 3.0 §3.1):

| Strategy | Method | Example |
|----------|--------|---------|
| **BM25** | Lexically similar chunks from a different document | "Porosity of Tyumen Suite" vs "Porosity of Vasyugan Suite" |
| **Cross-domain** | Semantically similar chunks from a different sub-domain (dense retrieval) | Reservoir query → negative from ore geology |
| **Adversarial** | LLM paraphrase with intentional key fact changes | "depth 2450 m" → "depth 3800 m" |

Adversarial generation is applied to 30% of train chunks (`--adversarial_ratio`).

**Balancing**: no more than 20% of triplets from any single sub-domain (`balance_by_subdomain`).

### Target Dataset Volumes

| Split | Minimum | Target |
|-------|---------|--------|
| Train (triplets) | 5,000 | 30,000–50,000 |
| Test (queries) | 200 | 500 |

---

## Training (step 5)

### Loss Functions

Main training uses two losses simultaneously:

| Loss | Purpose |
|------|---------|
| **MultipleNegativesRankingLoss** | In-batch negatives; scales with batch_size; analogous to Contrastive Loss from X-Intelligence §3.1 |
| **CoSENTLoss** | Scored pairs (positive=1.0, negative=0.0); improves fine-grained ranking |

**Periodic Hard Negative Mining** (`--periodic_mining`): runs after the main training loop. The current model re-indexes the corpus and retrieves the hardest examples, which are then used for additional fine-tuning via **TripletLoss**.

**Dev set**: if no dev file is passed explicitly (`--dev`), the script automatically reserves 10% of random triplets as a dev split for monitoring NDCG@10 during training.

### Default Hyperparameters

| Parameter | Value |
|-----------|-------|
| Base model | BAAI/bge-m3 |
| Batch size | 32 |
| Learning rate | 2e-5, cosine decay |
| Warmup | 10% of total steps |
| Epochs | 3 |
| Mixed precision | fp16 (CUDA only) |
| Eval / save | every ¼ epoch |
| Best checkpoint | by `NDCG@10` on dev |

### MLflow

If `MLFLOW_TRACKING_URI` is set, all parameters and metrics are logged automatically. If not set, logging is silently disabled.

---

## Evaluation (step 6)

Evaluation is performed on the **test set** — chunks from documents the model never saw during training.

| Metric | Target | What it measures |
|--------|--------|-----------------|
| **NDCG@10** | > 0.72 | Positional ranking quality (log-discount) |
| **Recall@5** | > 0.75 | Fraction of queries with positive in top-5 |
| **Recall@10** | > 0.85 | Fraction of queries with positive in top-10 |
| **MRR** | > 0.68 | Mean 1/rank of first relevant result |
| **ARR@5** | > 0.80 | Fraction of queries with at least 1 relevant in top-5 |

The primary metric for checkpoint selection is **NDCG@10**.

### Experiment Progress (E0–E4)

| Exp. | Change | Expected NDCG@10 |
|------|--------|-----------------|
| E0 — Baseline | BAAI/bge-m3 without fine-tuning | ~0.52 |
| E1 — +BM25 negatives | Fine-tuning on BM25 triplets | ~0.58 (+0.06) |
| E2 — +Cross-domain | Adding cross-domain negatives | ~0.63 (+0.05) |
| E3 — +Adversarial | Adding adversarial negatives | ~0.67 (+0.04) |
| E4 — +Periodic Mining | `--periodic_mining` | ~0.73 (+0.06) |

---

## Configuration (.env)

```bash
OPENAI_API_KEY=sk-...
OPENAI_API_BASE=https://openrouter.ai/api/v1

DOMAIN_DETECTION_MODEL=nvidia/nemotron-3-nano-30b-a3b  # step 1
QUERY_MODEL=anthropic/claude-haiku-4.5                 # step 2
ADVERSARIAL_MODEL=openai/gpt-4o-mini                   # step 4

MODEL_PATH=/models/finetuned-bge-m3-geo

MLFLOW_TRACKING_URI=http://172.17.0.1:5000
MLFLOW_EXPERIMENT_NAME=geo-embedder

HF_TOKEN=hf_...
# HTTPS_PROXY=socks5://172.17.0.1:1080
```

---

## Quick Start

```bash
mkdir -p raw_docs data models output
cp .env.example .env       # fill in your keys
cp your_docs/*.pdf raw_docs/

make build
make up                    # start pipeline container

make check-gpu             # verify CUDA
make shell                 # enter container

# Pre-run LLM cost estimate:
python scripts/estimate_price.py --num_chunks 1000

# Full pipeline in one command:
make run-pipeline
```

### Manual Step-by-Step (inside container)

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

## Makefile Commands

```bash
make build           # build Docker image
make up              # start pipeline container
make down            # stop containers
make shell           # enter container
make run-pipeline    # full pipeline (01→02→00→03→04→05)
make eval-baseline   # evaluate baseline BAAI/bge-m3
make eval-finetuned  # evaluate fine-tuned model
make check-gpu       # verify CUDA inside container
```

> **Note**: the `serve` service in `docker-compose.yml` is commented out.
> To enable serving, uncomment the `serve` section and use `make up-serve`.

---

## Docker Image Notes

Base image: `pytorch:2.1.0-cuda12.1-cudnn8-runtime`

Dockerfile adds: `poppler-utils` (PDF), `libxml2-dev` (DOCX), `build-essential` (rank-bm25), non-root user `geouser`, `HF_HOME` pointing to the mounted volume.

`cudnn8-runtime` is sufficient for fine-tuning via PyTorch. `-devel` is only needed when compiling CUDA extensions from source.

Volume mounts at runtime:

| Host | Container | Mode |
|------|-----------|------|
| `./raw_docs` | `/data/raw_docs` | ro |
| `./data` | `/data/processed` | rw |
| `./models` | `/models` | rw |
| `./output` | `/output` | rw |
| `./scripts` | `/app/scripts` | rw (hot-reload) |
| `./training` | `/app/training` | rw (hot-reload) |

> `utils/` is not mounted — it is baked into the image at `make build`.

---

## Connecting to Open WebUI

```
Admin Panel → Settings → RAG:
  Embedding Model Engine  : OpenAI
  Embedding Base URL      : http://<server_ip>:8080
  Embedding Model         : geo-embedder
  API Key                 : dummy
```

Server endpoints: `POST /v1/embeddings`, `GET /v1/models`, `GET /health`.
