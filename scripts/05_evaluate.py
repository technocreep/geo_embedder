"""
05_evaluate.py
==============
Оценка качества эмбеддера на геологическом тестовом сете.
Метрики (аналог X-Intelligence 3.0 §4, §5.2):
  - NDCG@5, NDCG@10
  - Recall@5, Recall@10
  - MRR (Mean Reciprocal Rank)
  - Acceptable Retrieval Rate (ARR) — доля запросов с хотя бы 1 релевантным в top-5

Запуск:
    # Сравнить baseline vs fine-tuned:
    python 05_evaluate.py \
        --test test_pairs.jsonl \
        --chunks chunks.jsonl \
        --models "BAAI/bge-m3" "./finetuned-bge-m3-geo" \
        --labels "Baseline" "Fine-tuned" \
        --output eval_results.json
"""

import argparse
import json
import math
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
from tqdm import tqdm


def load_test_data(path: Path):
    """
    Загружает тестовые пары.
    Формат: {"query": "...", "positive_chunk_id": "...", "positive_text": "..."}
    """
    queries, positives = {}, {}
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            qid = f"q{i}"
            queries[qid] = rec["query"]
            positives[qid] = rec["positive_chunk_id"]
    return queries, positives


def load_corpus(chunks_path: Path):
    """Загружает корпус чанков."""
    corpus = {}
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            corpus[c["id"]] = c["text"]
    return corpus


def encode_all(model, texts: list[str], batch_size: int = 64) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    embs = model.encode(
        texts, batch_size=batch_size,
        normalize_embeddings=True, show_progress_bar=True
    )
    return embs


def dcg(relevances: list[int], k: int) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))


def ndcg(relevances: list[int], k: int) -> float:
    ideal = sorted(relevances, reverse=True)
    ideal_dcg = dcg(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg(relevances[:k], k) / ideal_dcg


def evaluate_model(
    model_name: str,
    queries: dict,
    positives: dict,
    corpus: dict,
    top_k: int = 10,
    batch_size: int = 64,
) -> dict:
    from sentence_transformers import SentenceTransformer

    print(f"\n{'='*60}")
    print(f"Модель: {model_name}")
    print(f"{'='*60}")
    t0 = time.time()

    model = SentenceTransformer(model_name)

    qids = list(queries.keys())
    query_texts = [queries[qid] for qid in qids]
    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[cid] for cid in corpus_ids]

    print(f"Кодируем {len(query_texts)} вопросов...")
    q_embs = encode_all(model, query_texts, batch_size)

    print(f"Кодируем {len(corpus_texts)} чанков...")
    c_embs = encode_all(model, corpus_texts, batch_size)

    # Косинусное сходство
    scores = q_embs @ c_embs.T   # (n_queries, n_corpus)

    metrics_by_query = []
    subdomain_metrics = defaultdict(list)

    for i, qid in enumerate(qids):
        pos_id = positives[qid]
        ranked_idx = list(reversed(np.argsort(scores[i])))[:top_k]
        ranked_ids = [corpus_ids[idx] for idx in ranked_idx]

        # Relevance vector
        rel = [1 if cid == pos_id else 0 for cid in ranked_ids]

        # NDCG@5, NDCG@10
        ndcg5 = ndcg(rel, 5)
        ndcg10 = ndcg(rel, 10)

        # Recall@5, Recall@10
        recall5 = 1 if pos_id in ranked_ids[:5] else 0
        recall10 = 1 if pos_id in ranked_ids[:10] else 0

        # MRR
        mrr = 0.0
        for rank, cid in enumerate(ranked_ids, 1):
            if cid == pos_id:
                mrr = 1.0 / rank
                break

        # Acceptable Retrieval Rate (ARR): positive в top-5
        arr = recall5

        q_metrics = {
            "qid": qid,
            "ndcg5": ndcg5, "ndcg10": ndcg10,
            "recall5": recall5, "recall10": recall10,
            "mrr": mrr, "arr": arr,
        }
        metrics_by_query.append(q_metrics)

    # Агрегация
    def mean(vals):
        return float(np.mean(vals)) if vals else 0.0

    results = {
        "model": model_name,
        "n_queries": len(qids),
        "n_corpus": len(corpus_ids),
        "time_sec": round(time.time() - t0, 1),
        "NDCG@5":    mean([m["ndcg5"]   for m in metrics_by_query]),
        "NDCG@10":   mean([m["ndcg10"]  for m in metrics_by_query]),
        "Recall@5":  mean([m["recall5"] for m in metrics_by_query]),
        "Recall@10": mean([m["recall10"]for m in metrics_by_query]),
        "MRR":       mean([m["mrr"]     for m in metrics_by_query]),
        "ARR@5":     mean([m["arr"]     for m in metrics_by_query]),
    }

    print(f"\nРезультаты:")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:15s}: {v:.4f}")
        else:
            print(f"  {k:15s}: {v}")

    return results


def print_comparison_table(all_results: list[dict], labels: list[str]):
    """Печатает сравнительную таблицу (аналог таблиц 2-5 в X-Intelligence)."""
    metrics = ["NDCG@5", "NDCG@10", "Recall@5", "Recall@10", "MRR", "ARR@5"]

    print("\n" + "="*80)
    print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("="*80)

    # Заголовок
    header = f"{'Метрика':15s}" + "".join(f"{lb:20s}" for lb in labels) + "  Delta"
    print(header)
    print("-"*80)

    for m in metrics:
        vals = [r[m] for r in all_results]
        baseline = vals[0]
        row = f"{m:15s}" + "".join(f"{v:.4f}  ({v*100:.1f}%)    " for v in vals)
        if len(vals) > 1:
            delta = vals[-1] - vals[0]
            sign = "+" if delta >= 0 else ""
            row += f"  {sign}{delta:.4f}"
        print(row)

    print("="*80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True,
                        help="Список моделей для сравнения")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Метки для таблицы (по умолчанию = имена моделей)")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("eval_results.json"))
    args = parser.parse_args()

    labels = args.labels or args.models

    print("Загружаем тестовые данные...")
    queries, positives = load_test_data(args.test)
    corpus = load_corpus(args.chunks)
    print(f"Запросов: {len(queries)}, чанков в корпусе: {len(corpus)}")

    all_results = []
    for model_name in args.models:
        res = evaluate_model(
            model_name, queries, positives, corpus,
            args.top_k, args.batch_size
        )
        all_results.append(res)

    print_comparison_table(all_results, labels)

    with args.output.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Результаты сохранены: {args.output}")


if __name__ == "__main__":
    main()
