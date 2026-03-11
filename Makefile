# ─────────────────────────────────────────────────────────────────────────────
# Geo Embedder — Makefile
# Удобные команды для работы с контейнером
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: build up down shell run-pipeline eval-baseline eval-finetuned serve logs check-gpu

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
# Честный бенчмарк: тестовые документы отделены до обучения (нет data leakage).
#
# Структура /data/processed/ после запуска:
#   all_chunks.jsonl      — все чанки
#   all_queries.jsonl     — все сгенерированные вопросы
#   train_chunks.jsonl    — 80% документов (обучение)
#   test_chunks.jsonl     — 20% документов (оценка, модель их не видела)
#   train_queries.jsonl   — вопросы к train-чанкам
#   test_queries.jsonl    — вопросы к test-чанкам (честный тест-сет)
#   training_triplets.jsonl
#   dev_pairs.jsonl
run-pipeline:
	@echo "==> Шаг 1: Чанкование документов..."
	docker compose exec pipeline python scripts/01_chunk_documents.py \
		--input_dir /data/raw_docs \
		--output /data/processed/all_chunks.jsonl

	@echo "==> Шаг 2: Генерация вопросов ко всем чанкам..."
	docker compose exec pipeline python scripts/02_generate_queries.py \
		--chunks /data/processed/all_chunks.jsonl \
		--output /data/processed/all_queries.jsonl \
		--model anthropic/claude-haiku-4.5 \
		--queries_per_chunk 3 \
		--concurrency 10

	@echo "==> Шаг 3: Разделение на train/test по документам (20% — тест)..."
	docker compose exec pipeline python scripts/00_split_data.py \
		--chunks /data/processed/all_chunks.jsonl \
		--queries /data/processed/all_queries.jsonl \
		--output_dir /data/processed \
		--test_ratio 0.2 \
		--seed 42

	@echo "==> Шаг 4: Hard negative mining (только train)..."
	docker compose exec pipeline python scripts/03_mine_hard_negatives.py \
		--queries /data/processed/train_queries.jsonl \
		--chunks /data/processed/train_chunks.jsonl \
		--output /data/processed/training_triplets.jsonl \
		--strategy all \
		--adversarial_concurrency 10

	@echo "==> Шаг 5: Обучение (только train)..."
	docker compose exec pipeline python training/04_train_embedder.py \
		--triplets /data/processed/training_triplets.jsonl \
		--dev /data/processed/dev_pairs.jsonl \
		--output_dir /models/finetuned-bge-m3-geo \
		--epochs 3

	@echo "==> Шаг 6: Оценка baseline vs fine-tuned (на честном test-сете)..."
	docker compose exec pipeline python scripts/05_evaluate.py \
		--test /data/processed/test_queries.jsonl \
		--chunks /data/processed/test_chunks.jsonl \
		--models "/models/finetuned-bge-m3-geo" "BAAI/bge-m3" "google/embeddinggemma-300m" "yasserrmd/geo-gemma-300m-emb" "ai-forever/sbert_large_nlu_ru" "intfloat/multilingual-e5-large"\
		--labels "Fine-tuned" "Baseline" "google-gemma" "geo-gemma" "sberai-model" "infloat-e5-large"\
		--output /output/eval_results.json

	@echo "✓ Пайплайн завершён. Результаты: /output/eval_results.json"

# ── Только baseline оценка на test-сете (без обучения) ───────────────────────
eval-baseline:
	docker compose exec pipeline python scripts/05_evaluate.py \
		--test /data/processed/test_queries.jsonl \
		--chunks /data/processed/test_chunks.jsonl \
		--models "BAAI/bge-m3" \
		--labels "Baseline" \
		--output /output/eval_baseline.json

# ── Оценка дообученной модели (если обучение уже прошло) ─────────────────────
eval-finetuned:
	docker compose exec pipeline python scripts/05_evaluate.py \
		--test /data/processed/test_queries.jsonl \
		--chunks /data/processed/test_chunks.jsonl \
		--models "BAAI/bge-m3" "/models/finetuned-bge-m3-geo" \
		--labels "Baseline" "Fine-tuned" \
		--output /output/eval_results.json

# ── Проверить serving endpoint ────────────────────────────────────────────────
test-serve:
	curl -s -X POST http://localhost:8080/v1/embeddings \
		-H "Content-Type: application/json" \
		-d '{"input": ["юрские отложения Западной Сибири"], "model": "geo-embedder"}' \
		| python -c "import sys,json; d=json.load(sys.stdin); print('dim:', len(d['data'][0]['embedding']))"
