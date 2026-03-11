"""
00_split_data.py
================
Разделение данных на train/test по исходным документам.
Разделение по документам (не по чанкам) гарантирует отсутствие утечки данных:
тестовые чанки никогда не видит модель во время обучения.

Запуск:
    python 00_split_data.py \
        --chunks /data/processed/all_chunks.jsonl \
        --queries /data/processed/all_queries.jsonl \
        --output_dir /data/processed \
        --test_ratio 0.2 \
        --seed 42
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=Path, required=True,
                        help="Все чанки (all_chunks.jsonl)")
    parser.add_argument("--queries", type=Path, required=True,
                        help="Все запросы (all_queries.jsonl)")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Директория для результатов")
    parser.add_argument("--test_ratio", type=float, default=0.2,
                        help="Доля документов в тест-сете (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Загружаем чанки и группируем по источнику (документу)
    all_chunks = []
    with args.chunks.open(encoding="utf-8") as f:
        for line in f:
            all_chunks.append(json.loads(line))

    by_source: dict[str, list] = defaultdict(list)
    for chunk in all_chunks:
        source = chunk["metadata"].get("source", chunk["id"])
        by_source[source].append(chunk)

    sources = list(by_source.keys())
    random.shuffle(sources)

    n_test = max(1, int(len(sources) * args.test_ratio))
    test_sources = set(sources[:n_test])
    train_sources = set(sources[n_test:])

    train_chunks = [c for c in all_chunks if c["metadata"].get("source", c["id"]) in train_sources]
    test_chunks  = [c for c in all_chunks if c["metadata"].get("source", c["id"]) in test_sources]

    print(f"Документов всего: {len(sources)}")
    print(f"  train: {len(train_sources)} документов, {len(train_chunks)} чанков")
    print(f"  test:  {len(test_sources)} документов, {len(test_chunks)} чанков")

    # Загружаем запросы и разбиваем по тем же chunk_id
    train_chunk_ids = {c["id"] for c in train_chunks}
    test_chunk_ids  = {c["id"] for c in test_chunks}

    train_queries, test_queries = [], []
    with args.queries.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            cid = rec["positive_chunk_id"]
            if cid in train_chunk_ids:
                train_queries.append(rec)
            elif cid in test_chunk_ids:
                test_queries.append(rec)

    print(f"\nЗапросов всего: {len(train_queries) + len(test_queries)}")
    print(f"  train: {len(train_queries)}")
    print(f"  test:  {len(test_queries)}")

    # Сохраняем
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def write_jsonl(path: Path, records: list):
        with path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    write_jsonl(args.output_dir / "train_chunks.jsonl",  train_chunks)
    write_jsonl(args.output_dir / "test_chunks.jsonl",   test_chunks)
    write_jsonl(args.output_dir / "train_queries.jsonl", train_queries)
    write_jsonl(args.output_dir / "test_queries.jsonl",  test_queries)

    print(f"\n✓ Сохранено в {args.output_dir}:")
    print(f"  train_chunks.jsonl  ({len(train_chunks)} записей)")
    print(f"  test_chunks.jsonl   ({len(test_chunks)} записей)")
    print(f"  train_queries.jsonl ({len(train_queries)} записей)")
    print(f"  test_queries.jsonl  ({len(test_queries)} записей)")


if __name__ == "__main__":
    main()
