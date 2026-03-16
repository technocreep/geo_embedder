"""
03_mine_hard_negatives.py
=========================
Построение hard negatives для обучения эмбеддера.
Реализует все три стратегии из X-Intelligence 3.0 §3.1:
  1. BM25 Hard Negatives  — лексически похожие, семантически иные чанки
  2. Cross-domain Semantic — похожие чанки из другой подобласти геологии
  3. Adversarial           — LLM-перефразировка с изменением ключевых фактов

Выход: training_triplets.jsonl  (query, positive, negative)
       для SentenceTransformers InputExample(texts=[q, pos, neg])

Запуск:
    python 03_mine_hard_negatives.py \
        --queries queries.jsonl \
        --chunks chunks.jsonl \
        --output training_triplets.jsonl \
        --model BAAI/bge-m3 \
        --strategy all
"""

import argparse
import asyncio
import json
import os
import random
from pathlib import Path
from typing import Optional
from collections import defaultdict

import httpx
from openai import AsyncOpenAI
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()


def get_device(override: Optional[str] = None) -> str:
    import torch
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ─── Стратегия 1: BM25 Hard Negatives ────────────────────────────────────────

def build_bm25_negatives(
    queries: list[dict],
    all_chunks: list[dict],
    top_k: int = 20,
) -> list[Optional[str]]:
    """
    Для каждого запроса: найти чанки с высоким BM25-score (лексически похожие),
    но из другого документа (не true positive). Это hard negatives: модель может
    их «перепутать» с правильным ответом по ключевым словам.
    """
    from rank_bm25 import BM25Okapi

    corpus_ids = [c["id"] for c in all_chunks]
    corpus_texts = [c["text"] for c in all_chunks]
    tokenized = [t.lower().split() for t in corpus_texts]
    bm25 = BM25Okapi(tokenized)

    negatives = []
    for query_rec in tqdm(queries, desc="BM25 hard negatives"):
        query_tokens = query_rec["query"].lower().split()
        scores = bm25.get_scores(query_tokens)

        ranked = sorted(enumerate(scores), key=lambda x: -x[1])
        neg = None
        for idx, score in ranked[:top_k]:
            cid = corpus_ids[idx]
            if cid != query_rec["positive_chunk_id"] and score > 0:
                neg = corpus_texts[idx]
                break
        negatives.append(neg)

    return negatives


# ─── Стратегия 2: Cross-domain Semantic Negatives ────────────────────────────

def build_crossdomain_negatives(
    queries: list[dict],
    all_chunks: list[dict],
    model_name: str = "BAAI/bge-m3",
    top_k: int = 5,
    batch_size: int = 64,
    device: Optional[str] = None,
) -> list[Optional[str]]:
    """
    Семантически похожие чанки из ДРУГОЙ геологической подобласти.
    Например: запрос о нефтяных коллекторах → negative из рудной геологии.
    """
    from sentence_transformers import SentenceTransformer

    selected_device = get_device(device)
    print(f"Используем устройство: {selected_device}")
    print(f"Загружаем модель {model_name} для cross-domain negatives...")
    model = SentenceTransformer(model_name, device=selected_device)

    by_subdomain: dict[str, list[dict]] = defaultdict(list)
    for c in all_chunks:
        sd = c["metadata"].get("subdomain", "общая_геология")
        by_subdomain[sd].append(c)

    print("Кодируем корпус...")
    all_texts = [c["text"] for c in all_chunks]
    all_ids = [c["id"] for c in all_chunks]
    all_subdomains = [c["metadata"].get("subdomain", "общая_геология") for c in all_chunks]

    corpus_embs = model.encode(
        all_texts, batch_size=batch_size,
        normalize_embeddings=True, show_progress_bar=True
    )

    query_texts = [q["query"] for q in queries]
    query_embs = model.encode(
        query_texts, batch_size=batch_size,
        normalize_embeddings=True, show_progress_bar=True
    )

    import numpy as np
    scores_matrix = query_embs @ corpus_embs.T  # (n_queries, n_corpus)

    negatives = []
    for i, query_rec in enumerate(queries):
        query_subdomain = query_rec.get("subdomain", "")
        pos_id = query_rec["positive_chunk_id"]

        ranked = np.argsort(-scores_matrix[i])
        neg = None
        for idx in ranked[:50]:
            cid = all_ids[idx]
            csd = all_subdomains[idx]
            if cid != pos_id and csd != query_subdomain:
                neg = all_texts[idx]
                break
        negatives.append(neg)

    return negatives


# ─── Стратегия 3: Adversarial Negatives (LLM-перефразировка) ─────────────────

ADVERSARIAL_PROMPT = """Ты эксперт-геолог. Перефразируй следующий геологический текст так, чтобы:
1. Текст звучал правдоподобно и профессионально
2. Но содержал искажение ключевых фактов: измени числа (глубины, концентрации, возраст пород),
   названия формаций/горизонтов, типы пород или геохимические показатели
3. Объём — примерно тот же
4. Верни ТОЛЬКО перефразированный текст, без пояснений

Исходный текст:
{text}"""


async def build_adversarial_negatives(
    chunks: list[dict],
    model: str,
    sample_ratio: float = 0.3,
    concurrency: int = 10,
) -> dict[str, str]:
    """
    Генерирует adversarial negative для случайной выборки чанков.
    Возвращает dict: chunk_id -> adversarial_text
    """
    sample = random.sample(chunks, int(len(chunks) * sample_ratio))

    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    http_client = httpx.AsyncClient(proxy=proxy) if proxy else None
    client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_API_BASE", None),
        http_client=http_client,
    )

    sem = asyncio.Semaphore(concurrency)
    result: dict[str, str] = {}
    pbar = tqdm(total=len(sample), desc="Adversarial negatives")

    async def process_one(chunk):
        async with sem:
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user",
                               "content": ADVERSARIAL_PROMPT.format(text=chunk["text"][:1500])}],
                    temperature=0.8,
                    max_tokens=600,
                )
                adv_text = resp.choices[0].message.content.strip()
                if len(adv_text) > 50:
                    result[chunk["id"]] = adv_text
            except Exception as e:
                print(f"\n[error] {chunk['id']}: {e}")
            finally:
                pbar.update(1)

    await asyncio.gather(*[process_one(c) for c in sample])
    pbar.close()
    await client.close()
    if http_client:
        await http_client.aclose()

    return result


# ─── Сборка итогового датасета триплетов ─────────────────────────────────────

def build_triplets(
    queries: list[dict],
    bm25_negs: list[Optional[str]],
    crossdomain_negs: list[Optional[str]],
    adversarial_negs: dict[str, str],  # chunk_id -> text
) -> list[dict]:
    """
    Объединяет все негативы. Каждый запрос может дать до 3 триплетов
    (по одному от каждой стратегии).
    """
    triplets = []
    for i, query_rec in enumerate(queries):
        q = query_rec["query"]
        pos = query_rec["positive_text"]
        pos_id = query_rec["positive_chunk_id"]
        subdomain = query_rec.get("subdomain", "")

        if bm25_negs[i]:
            triplets.append({
                "query": q, "positive": pos, "negative": bm25_negs[i],
                "neg_type": "bm25", "subdomain": subdomain
            })
        if crossdomain_negs[i]:
            triplets.append({
                "query": q, "positive": pos, "negative": crossdomain_negs[i],
                "neg_type": "crossdomain", "subdomain": subdomain
            })
        if pos_id in adversarial_negs:
            triplets.append({
                "query": q, "positive": pos, "negative": adversarial_negs[pos_id],
                "neg_type": "adversarial", "subdomain": subdomain
            })

    return triplets


def balance_by_subdomain(triplets: list[dict], max_per_domain_ratio: float = 0.20) -> list[dict]:
    """
    Балансировка: не более max_per_domain_ratio от общего объёма на одну подобласть.
    Аналог subdomain distribution filter из X-Intelligence 3.0 §2.2.
    """
    by_domain: dict[str, list] = defaultdict(list)
    for t in triplets:
        by_domain[t["subdomain"]].append(t)

    max_per = int(len(triplets) * max_per_domain_ratio)
    balanced = []
    for domain, items in by_domain.items():
        if len(items) > max_per:
            items = random.sample(items, max_per)
        balanced.extend(items)

    random.shuffle(balanced)
    return balanced


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, default=Path("queries.jsonl"))
    parser.add_argument("--chunks", type=Path, default=Path("chunks.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("training_triplets.jsonl"))
    parser.add_argument("--model", default="BAAI/bge-m3",
                        help="Модель для cross-domain semantic negatives")
    parser.add_argument("--strategy", choices=["all", "bm25", "crossdomain", "adversarial"],
                        default="all")

    parser.add_argument("--adversarial_ratio", type=float, default=0.3,
                        help="Доля чанков для adversarial генерации")
    parser.add_argument("--adversarial_concurrency", type=int, default=10,
                        help="Число параллельных запросов к API для adversarial")
    parser.add_argument("--device", default=None,
                        help="Устройство для эмбеддингов: cuda/mps/cpu (по умолчанию — автовыбор)")
    parser.add_argument("--balance", action="store_true", default=True,
                        help="Балансировка по подобластям")
    args = parser.parse_args()

    queries = [json.loads(l) for l in args.queries.open(encoding="utf-8")]
    all_chunks = [json.loads(l) for l in args.chunks.open(encoding="utf-8")]
    print(f"Запросов: {len(queries)}, чанков в корпусе: {len(all_chunks)}")

    bm25_negs = [None] * len(queries)
    if args.strategy in ("all", "bm25"):
        bm25_negs = build_bm25_negatives(queries, all_chunks)

    crossdomain_negs = [None] * len(queries)
    if args.strategy in ("all", "crossdomain"):
        crossdomain_negs = build_crossdomain_negatives(queries, all_chunks, args.model, device=args.device)

    model = os.environ["ADVERSARIAL_MODEL"]
    adversarial_negs = {}
    if args.strategy in ("all", "adversarial"):
        adversarial_negs = asyncio.run(build_adversarial_negatives(
            all_chunks,
            model=model,
            sample_ratio=args.adversarial_ratio,
            concurrency=args.adversarial_concurrency,
        ))

    triplets = build_triplets(queries, bm25_negs, crossdomain_negs, adversarial_negs)
    print(f"\nТриплетов до балансировки: {len(triplets)}")

    if args.balance:
        triplets = balance_by_subdomain(triplets)
        print(f"Триплетов после балансировки: {len(triplets)}")

    neg_types = defaultdict(int)
    for t in triplets:
        neg_types[t["neg_type"]] += 1
    print("\nРаспределение по типу негативов:")
    for k, v in sorted(neg_types.items()):
        print(f"  {k:20s} {v:5d}")

    with args.output.open("w", encoding="utf-8") as f:
        for t in triplets:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"\n✓ Сохранено: {args.output}")


if __name__ == "__main__":
    main()
