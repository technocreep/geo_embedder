# ─────────────────────────────────────────────────────────────────────────────
# Geo Embedder — Makefile
# Удобные команды для работы с контейнером
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: build up down shell run-pipeline serve logs check-gpu

# ── Сборка образа ─────────────────────────────────────────────────────────────
build:
	docker compose build

hard-restart:
	docker compose down
	docker compose build --no-cache
	docker compose up -d pipeline

# ── Поднять все сервисы (pipeline + serve) ────────────────────────────────────
up:
	docker compose up -d pipeline serve

# ── Только сервер эмбеддера (если модель уже обучена) ────────────────────────
up-serve:
	docker compose up -d serve

# ── Остановить всё ────────────────────────────────────────────────────────────
down:
	docker compose down

# ── Войти в контейнер pipeline ────────────────────────────────────────────────
shell:
	docker compose exec pipeline bash

# ── Проверить GPU внутри контейнера ──────────────────────────────────────────
check-gpu:
	docker compose exec pipeline python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

# ── Логи serve ────────────────────────────────────────────────────────────────
logs:
	docker compose logs -f serve

# ─── Пайплайн: запустить все шаги последовательно ───────────────────────────
run-pipeline:
	@echo "==> Шаг 1: Чанкование документов..."
	docker compose exec pipeline python scripts/01_chunk_documents.py \
		--input_dir /data/raw_docs \
		--output /data/processed/chunks.jsonl

	@echo "==> Шаг 2: Генерация вопросов..."
	docker compose exec pipeline python scripts/02_generate_queries.py \
		--chunks /data/processed/chunks.jsonl \
		--output /data/processed/queries.jsonl \
		--model gpt-4o-mini \
		--queries_per_chunk 3 \
		--concurrency 10

	@echo "==> Шаг 3: Hard negative mining..."
	docker compose exec pipeline python scripts/03_mine_hard_negatives.py \
		--queries /data/processed/queries.jsonl \
		--chunks /data/processed/chunks.jsonl \
		--output /data/processed/training_triplets.jsonl \
		--strategy all \
		--adversarial_concurrency 10

	@echo "==> Шаг 4: Обучение..."
	docker compose exec pipeline python training/04_train_embedder.py \
		--triplets /data/processed/training_triplets.jsonl \
		--dev /data/processed/dev_pairs.jsonl \
		--output_dir /models/finetuned-bge-m3-geo \
		--epochs 3

	@echo "==> Шаг 5: Оценка baseline vs fine-tuned..."
	docker compose exec pipeline python scripts/05_evaluate.py \
		--test /data/processed/test_pairs.jsonl \
		--chunks /data/processed/chunks.jsonl \
		--models "BAAI/bge-m3" "/models/finetuned-bge-m3-geo" \
		--labels "Baseline" "Fine-tuned" \
		--output /output/eval_results.json

	@echo "✓ Пайплайн завершён. Результаты: ./output/eval_results.json"

# ── Только baseline оценка (быстрая проверка без обучения) ───────────────────
eval-baseline:
	docker compose exec pipeline python scripts/05_evaluate.py \
		--test /data/processed/test_pairs.jsonl \
		--chunks /data/processed/chunks.jsonl \
		--models "BAAI/bge-m3" \
		--labels "Baseline" \
		--output /output/eval_baseline.json

# ── Проверить serving endpoint ────────────────────────────────────────────────
test-serve:
	curl -s -X POST http://localhost:8080/v1/embeddings \
		-H "Content-Type: application/json" \
		-d '{"input": ["юрские отложения Западной Сибири"], "model": "geo-embedder"}' \
		| python -c "import sys,json; d=json.load(sys.stdin); print('dim:', len(d['data'][0]['embedding']))"
