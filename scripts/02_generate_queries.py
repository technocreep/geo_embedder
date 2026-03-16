"""
02_generate_queries.py
======================
Генерация вопросов к геологическим чанкам с помощью LLM.
Аналог X-Intelligence 3.0 §2.1 «Domain-Specific Question Construction» и §3.3 «Query Generation».

Запуск:
    python 02_generate_queries.py \
        --chunks chunks.jsonl \
        --output queries.jsonl \
        --queries_per_chunk 3 \
        --concurrency 10
"""

import argparse
import asyncio
import json
import os
from pathlib import Path
from dotenv import load_dotenv

import httpx
from openai import AsyncOpenAI
from tqdm import tqdm

load_dotenv()

# ─── Системный промпт для генерации геологических вопросов ───────────────────
SYSTEM_PROMPT = """Ты эксперт-геолог с 20-летним опытом в области геологоразведки и полезных ископаемых.
Твоя задача — формулировать вопросы к геологическим текстам для системы поиска информации (RAG).

Правила формулировки вопросов:
1. Вопрос должен быть конкретным — ответ на него содержится именно в данном тексте
2. Используй профессиональную геологическую терминологию
3. Вопрос должен требовать содержательного ответа (не «да/нет»)
4. Формулируй вопросы так, как их задают геологи, буровые инженеры или аналитики недр
5. Варьируй стиль: одни вопросы — как в научной статье, другие — как в рабочем запросе специалиста

Верни ТОЛЬКО JSON-массив вопросов, без пояснений:
["вопрос 1", "вопрос 2", "вопрос 3"]"""

USER_PROMPT_TEMPLATE = """Текст из геологического документа (подобласть: {subdomain}):

---
{text}
---

Сформулируй {n} вопросов разного стиля и сложности, на которые данный текст является ответом."""


# ─── Асинхронный клиент ───────────────────────────────────────────────────────

def _make_async_client() -> tuple[AsyncOpenAI, httpx.AsyncClient | None]:
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    http_client = httpx.AsyncClient(proxy=proxy) if proxy else None
    client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_API_BASE", None),
        http_client=http_client,
    )
    return client, http_client


async def call_openai(
    client: AsyncOpenAI,
    text: str,
    subdomain: str,
    n: int,
    model: str,
) -> list[str]:
    prompt = USER_PROMPT_TEMPLATE.format(text=text[:2000], subdomain=subdomain, n=n)
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=512,
    )
    return _parse_json_list(resp.choices[0].message.content.strip())


def _parse_json_list(text: str) -> list[str]:
    """Извлекает JSON-массив из ответа LLM."""
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        result = json.loads(text[start:end + 1])
        return [q.strip() for q in result if isinstance(q, str) and len(q) > 10]
    except json.JSONDecodeError:
        return []


# ─── Фильтрация вопросов ──────────────────────────────────────────────────────

def filter_query(query: str) -> bool:
    """
    Фильтрует некачественные вопросы.
    Аналог X-Intelligence §5 (фильтрация невалидируемых и тривиальных вопросов).
    """
    if len(query) < 15:
        return False
    reject_patterns = [
        "что такое геология",
        "расскажи о",
        "опиши",
        r"^\s*что\s*\?",
    ]
    q_lower = query.lower()
    for pat in reject_patterns:
        if pat in q_lower:
            return False
    if "?" not in query and not any(w in q_lower for w in ["какой", "какая", "каков", "как", "где", "когда", "почему", "чем", "что"]):
        return False
    return True


# ─── Основной пайплайн ────────────────────────────────────────────────────────

async def _run(args):
    chunks = []
    with args.chunks.open(encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    if args.max_chunks:
        chunks = chunks[:args.max_chunks]
    
    model = os.environ["QUERY_MODEL"]
    print(f"Чанков для обработки: {len(chunks)}")
    print(f"Модель: {model}, параллелизм: {args.concurrency}")

    client, http_client = _make_async_client()
    sem = asyncio.Semaphore(args.concurrency)
    pbar = tqdm(total=len(chunks), desc="Генерация вопросов")
    total_queries = 0
    skipped = 0

    async def process_one(chunk) -> list[dict]:
        nonlocal skipped
        async with sem:
            try:
                queries = await call_openai(
                    client,
                    text=chunk["text"],
                    subdomain=chunk["metadata"].get("subdomain", "общая_геология"),
                    n=args.queries_per_chunk,
                    model=model,
                )
                queries = [q for q in queries if filter_query(q)]
            except Exception as e:
                print(f"\n[error] chunk {chunk['id']}: {e}")
                queries = []
            finally:
                pbar.update(1)

        if not queries:
            skipped += 1
            return []
        return [
            {
                "query": q,
                "positive_chunk_id": chunk["id"],
                "positive_text": chunk["text"],
                "subdomain": chunk["metadata"].get("subdomain"),
                "source": chunk["metadata"].get("source"),
            }
            for q in queries
        ]

    results = await asyncio.gather(*[process_one(c) for c in chunks])
    pbar.close()
    await client.close()
    if http_client:
        await http_client.aclose()

    with args.output.open("w", encoding="utf-8") as fout:
        for records in results:
            for rec in records:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total_queries += 1

    print(f"\n✓ Сгенерировано вопросов: {total_queries}")
    print(f"  Пропущено чанков: {skipped}")
    print(f"  Среднее вопросов/чанк: {total_queries / max(len(chunks) - skipped, 1):.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=Path, default=Path("chunks.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("queries.jsonl"))
    parser.add_argument("--queries_per_chunk", type=int, default=3)
    parser.add_argument("--max_chunks", type=int, default=None,
                        help="Ограничить кол-во чанков (для тестирования)")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Число параллельных запросов к API")
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
