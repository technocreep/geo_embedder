"""
04_train_embedder.py
====================
Дообучение BAAI/bge-m3 на геологическом корпусе.
Реализует полный цикл из X-Intelligence 3.0 §3.1:
  - MultipleNegativesRankingLoss (Contrast Loss аналог)
  - CoSENTLoss (дополнительный loss)
  - Periodic Hard Negative Mining (каждые N шагов)
  - Evaluation на dev-сете

Совместимость: SentenceTransformers >= 2.7, torch >= 2.0

Запуск:
    python 04_train_embedder.py \
        --triplets training_triplets.jsonl \
        --dev dev_pairs.jsonl \
        --output_dir ./finetuned-bge-m3-geo \
        --epochs 3
"""

import argparse
import json
import logging
import math
import os
import random
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import mlflow
import torch
from mlflow.tracking import MlflowClient
from torch.utils.data import DataLoader
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    evaluation,
    util as st_util,
)
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.trainer import SentenceTransformerTrainer
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_device(override: str | None = None) -> str:
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ─── Загрузка данных ──────────────────────────────────────────────────────────

def load_triplets(path: Path) -> list[InputExample]:
    """Загружает (query, positive, negative) триплеты."""
    examples = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            examples.append(InputExample(
                texts=[rec["query"], rec["positive"], rec["negative"]]
            ))
    return examples


def load_pairs(path: Path) -> list[InputExample]:
    """
    Загружает пары (query, positive) для MultipleNegativesRankingLoss.
    Формат строки: {"query": "...", "positive": "..."}
    """
    examples = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            examples.append(InputExample(
                texts=[rec["query"], rec["positive"]]
            ))
    return examples


def make_dev_from_triplets(triplets_path: Path, dev_path: Path, dev_ratio: float = 0.1) -> Path:
    """
    Создаёт dev_pairs.jsonl из случайного сплита training_triplets.jsonl.
    Возвращает путь к созданному файлу.
    """
    all_lines = triplets_path.read_text(encoding="utf-8").splitlines()
    random.shuffle(all_lines)
    n_dev = max(50, int(len(all_lines) * dev_ratio))
    dev_lines = all_lines[:n_dev]

    with dev_path.open("w", encoding="utf-8") as f:
        for line in dev_lines:
            t = json.loads(line)
            cid = f"chunk_{hash(t['positive']) & 0xFFFFFFFF}"
            f.write(json.dumps({
                "query": t["query"],
                "positive_chunk_id": cid,
                "positive_text": t["positive"],
            }, ensure_ascii=False) + "\n")

    logger.info(f"[dev split] Создан из {len(all_lines)} триплетов → {n_dev} dev-пар: {dev_path}")
    return dev_path


def load_dev_pairs(path: Path) -> tuple[list[str], list[str], list[float]]:
    """
    Загружает dev-сет для InformationRetrievalEvaluator.
    Формат: {"query": "...", "positive_chunk_id": "...", "positive_text": "..."}
    Возвращает queries, corpus, relevant_docs dict.
    """
    queries = {}
    corpus = {}
    relevant = {}

    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            qid = f"q{i}"
            cid = rec["positive_chunk_id"]
            queries[qid] = rec["query"]
            corpus[cid] = rec["positive_text"]
            relevant[qid] = {cid}

    return queries, corpus, relevant


# ─── Periodic Hard Negative Mining Callback ───────────────────────────────────

class HardNegativeMiningCallback:
    """
    Аналог §3.1 X-Intelligence: каждые `mine_every` шагов переиндексирует
    корпус текущей моделью и обновляет обучающий датасет harder negatives.

    В текущей реализации SentenceTransformers Trainer не поддерживает
    on-the-fly замену датасета — поэтому mining запускается между эпохами
    (проще и надёжнее для production).
    """

    def __init__(self, model: SentenceTransformer, corpus_chunks: list[dict],
                 queries: list[dict], mine_every_epochs: int = 1):
        self.model = model
        self.corpus_chunks = corpus_chunks
        self.queries = queries
        self.mine_every_epochs = mine_every_epochs
        self._epoch = 0

    def on_epoch_end(self) -> list[InputExample]:
        self._epoch += 1
        if self._epoch % self.mine_every_epochs != 0:
            return []

        logger.info(f"[HardNegativeMining] Эпоха {self._epoch}: переиндексация корпуса...")

        # Закодировать все чанки текущей моделью
        corpus_texts = [c["text"] for c in self.corpus_chunks]
        corpus_ids = [c["id"] for c in self.corpus_chunks]
        corpus_embs = self.model.encode(
            corpus_texts, batch_size=64,
            normalize_embeddings=True, show_progress_bar=False
        )

        new_examples = []
        query_texts = [q["query"] for q in self.queries]
        query_embs = self.model.encode(
            query_texts, batch_size=64,
            normalize_embeddings=True, show_progress_bar=False
        )

        import numpy as np
        scores = query_embs @ corpus_embs.T

        for i, q_rec in enumerate(self.queries):
            pos_id = q_rec["positive_chunk_id"]
            ranked = list(reversed(sorted(enumerate(scores[i]), key=lambda x: x[1])))

            # Самый похожий чанк, который не является positive — hard negative
            for idx, score in ranked[:20]:
                cid = corpus_ids[idx]
                if cid != pos_id:
                    new_examples.append(InputExample(
                        texts=[q_rec["query"], q_rec["positive_text"], corpus_texts[idx]]
                    ))
                    break

        logger.info(f"[HardNegativeMining] Собрано {len(new_examples)} новых hard negatives")
        return new_examples


# ─── Основная функция обучения ────────────────────────────────────────────────

def _setup_mlflow(args) -> tuple[bool, str]:
    """Настраивает MLflow. Возвращает (enabled, run_name)."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if not tracking_uri:
        logger.info("[MLflow] MLFLOW_TRACKING_URI не задан — логирование отключено")
        return False, ""

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    exp_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "geo-embedder")
    exp = client.get_experiment_by_name(exp_name)
    if exp is None:
        logger.info(f"[MLflow] Создаём эксперимент: {exp_name}")
        exp_id = client.create_experiment(name=exp_name)
    else:
        exp_id = exp.experiment_id

    mlflow.set_experiment(experiment_id=exp_id)
    mlflow.enable_system_metrics_logging()

    model_tag = args.base_model.split("/")[-1]
    run_name = f"{model_tag}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    logger.info(f"[MLflow] Эксперимент: {exp_name}, run: {run_name}, URI: {tracking_uri}")
    return True, run_name


def train(args):
    device = get_device(args.device)
    logger.info(f"Используем устройство: {device}")
    use_amp = device == "cuda"  # AMP работает только на CUDA

    # ── MLflow ──
    mlflow_enabled, run_name = _setup_mlflow(args)
    run_ctx = mlflow.start_run(run_name=run_name) if mlflow_enabled else nullcontext()

    with run_ctx:
        if mlflow_enabled:
            mlflow.log_params({
                "base_model": args.base_model,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "device": device,
                "triplets_file": str(args.triplets),
                "periodic_mining": args.periodic_mining,
            })

        # Загружаем базовую модель
        logger.info(f"Загружаем базовую модель: {args.base_model}")
        model = SentenceTransformer(args.base_model, device=device)

        # Загружаем данные
        triplets = load_triplets(args.triplets)
        logger.info(f"Обучающих триплетов: {len(triplets)}")
        if mlflow_enabled:
            mlflow.log_param("num_triplets", len(triplets))

        # Dev evaluator
        evaluator = None
        dev_path = Path(args.dev) if args.dev else None
        if dev_path and not dev_path.exists():
            # Автоматически создаём dev-сплит из триплетов
            dev_path = make_dev_from_triplets(args.triplets, dev_path, dev_ratio=0.1)
        if dev_path and dev_path.exists():
            queries, corpus, relevant = load_dev_pairs(dev_path)
            evaluator = evaluation.InformationRetrievalEvaluator(
                queries=queries,
                corpus=corpus,
                relevant_docs=relevant,
                name="geo-dev",
                score_functions={"cos_sim": st_util.cos_sim},
                batch_size=64,
            )
            logger.info(f"Dev evaluator: {len(queries)} вопросов, {len(corpus)} чанков")

        # ── Loss 1: TripletLoss (explicit hard negatives, используется в periodic mining) ──
        triplet_loss = losses.TripletLoss(model=model)

        # ── Loss 2: MultipleNegativesRankingLoss (in-batch negatives) ──
        # Для MNRL нужны только пары (query, positive)
        pairs = [InputExample(texts=[t.texts[0], t.texts[1]]) for t in triplets]
        train_dataloader_mnrl = DataLoader(pairs, shuffle=True, batch_size=args.batch_size)
        mnrl_loss = losses.MultipleNegativesRankingLoss(model=model)

        # ── Loss 3: CoSENTLoss — добавляем scored pairs ──
        # Positive пары получают score=1, negative pairs score=0
        cosent_examples = []
        for t in triplets:
            cosent_examples.append(InputExample(texts=[t.texts[0], t.texts[1]], label=1.0))
            cosent_examples.append(InputExample(texts=[t.texts[0], t.texts[2]], label=0.0))
        train_dataloader_cosent = DataLoader(cosent_examples, shuffle=True, batch_size=args.batch_size)
        cosent_loss = losses.CoSENTLoss(model=model)

        # Warmup steps
        total_steps = (len(train_dataloader_mnrl) * args.epochs)
        warmup_steps = math.ceil(total_steps * 0.1)

        logger.info(f"Всего шагов: {total_steps}, warmup: {warmup_steps}")
        logger.info(f"Выходная директория: {args.output_dir}")
        if mlflow_enabled:
            mlflow.log_params({"total_steps": total_steps, "warmup_steps": warmup_steps})

        # ── Обучение ──
        # Используем fit() с несколькими лоссами (weighted)
        model.fit(
            train_objectives=[
                (train_dataloader_mnrl, mnrl_loss),      # основной loss
                (train_dataloader_cosent, cosent_loss),  # дополнительный
            ],
            evaluator=evaluator,
            epochs=args.epochs,
            warmup_steps=warmup_steps,
            optimizer_params={"lr": args.lr},
            output_path=str(args.output_dir),
            save_best_model=evaluator is not None,
            evaluation_steps=max(len(train_dataloader_mnrl) // 4, 100) if evaluator else 0,
            show_progress_bar=True,
            use_amp=use_amp,   # fp16/bf16 mixed precision (только CUDA)
        )

        logger.info(f"✓ Модель сохранена: {args.output_dir}")

        # ── Periodic Hard Negative Mining (между эпохами) ──
        if args.periodic_mining and args.triplets.exists():
            logger.info("Запуск Periodic Hard Negative Mining...")
            # В production: загружать chunks.jsonl напрямую
            query_recs = [{"query": t.texts[0], "positive_text": t.texts[1],
                           "positive_chunk_id": f"chunk_{i}"}
                          for i, t in enumerate(triplets[:500])]
            corpus_chunks = [{"id": f"chunk_{i}", "text": t.texts[1]}
                             for i, t in enumerate(triplets[:500])]

            miner = HardNegativeMiningCallback(
                model=model,
                corpus_chunks=corpus_chunks,
                queries=query_recs,
            )
            new_examples = miner.on_epoch_end()
            if new_examples:
                logger.info(f"Дообучение на {len(new_examples)} hard negatives...")
                hn_dl = DataLoader(new_examples, shuffle=True, batch_size=args.batch_size)
                model.fit(
                    train_objectives=[(hn_dl, triplet_loss)],
                    epochs=1,
                    warmup_steps=50,
                    output_path=str(args.output_dir) + "_hn",
                    show_progress_bar=True,
                    use_amp=use_amp,
                )

        # ── Финальная оценка ──
        if evaluator:
            logger.info("Финальная оценка на dev-сете...")
            result = evaluator(model)
            logger.info(f"Dev результаты: {result}")
            if mlflow_enabled and isinstance(result, dict):
                safe_metrics = {
                    k.replace("@", "_at_"): float(v)
                    for k, v in result.items()
                    if isinstance(v, (int, float))
                }
                mlflow.log_metrics(safe_metrics)

        # ── Логируем артефакт модели ──
        if mlflow_enabled and args.output_dir.exists():
            mlflow.log_artifacts(str(args.output_dir), artifact_path="model")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--triplets", type=Path, required=True,
                        help="JSONL с триплетами (query, positive, negative)")
    parser.add_argument("--dev", type=str, default="",
                        help="JSONL для dev evaluation (query, positive_chunk_id, positive_text)")
    parser.add_argument("--output_dir", type=Path, default=Path("finetuned-bge-m3-geo"))
    parser.add_argument("--base_model", default="BAAI/bge-m3")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--periodic_mining", action="store_true", default=False,
                        help="Запустить hard negative mining после обучения")
    parser.add_argument("--device", default=None,
                        help="Устройство: cuda/mps/cpu (по умолчанию — автовыбор)")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
