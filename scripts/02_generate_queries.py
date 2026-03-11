"""
02_generate_queries.py
======================
Генерация вопросов к геологическим чанкам с помощью LLM.
Аналог X-Intelligence 3.0 §2.1 «Domain-Specific Question Construction» и §3.3 «Query Generation».

Поддерживаемые LLM-бэкенды (через переменную --backend):
  - openai   : GPT-4o-mini / GPT-4o  (требует OPENAI_API_KEY)
  - ollama   : локальная LLM через Ollama API

Запуск:
    python 02_generate_queries.py \
        --chunks chunks.jsonl \
        --output queries.jsonl \
        --backend openai \
        --model gpt-4o-mini \
        --queries_per_chunk 3
"""

import argparse
import json
import os
import time
import random
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

from tqdm import tqdm

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


# ─── Клиенты для разных бэкендов ─────────────────────────────────────────────

def call_openai(text: str, subdomain: str, n: int, model: str) -> list[str]:
    import httpx
    from openai import OpenAI
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    http_client = httpx.Client(proxy=proxy) if proxy else None
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_API_BASE", None),
        http_client=http_client,
    )
    prompt = USER_PROMPT_TEMPLATE.format(text=text[:2000], subdomain=subdomain, n=n)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=512,
    )
    content = resp.choices[0].message.content.strip()
    return _parse_json_list(content)


def call_ollama(text: str, subdomain: str, n: int, model: str) -> list[str]:
    import urllib.request
    prompt = USER_PROMPT_TEMPLATE.format(text=text[:2000], subdomain=subdomain, n=n)
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.7},
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        content = json.loads(resp.read())["message"]["content"].strip()
    return _parse_json_list(content)


def _parse_json_list(text: str) -> list[str]:
    """Извлекает JSON-массив из ответа LLM."""
    # Попытка найти [...] в тексте
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
    # Слишком общие вопросы
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
    # Должен содержать знак вопроса или быть именным запросом
    if "?" not in query and not any(w in q_lower for w in ["какой", "какая", "каков", "как", "где", "когда", "почему", "чем", "что"]):
        return False
    return True


# ─── Основной пайплайн ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=Path, default=Path("chunks.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("queries.jsonl"))
    parser.add_argument("--backend", choices=["openai", "ollama"], default="openai")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--queries_per_chunk", type=int, default=3)
    parser.add_argument("--max_chunks", type=int, default=None,
                        help="Ограничить кол-во чанков (для тестирования)")
    parser.add_argument("--sleep", type=float, default=0.5,
                        help="Пауза между запросами к API (сек)")
    args = parser.parse_args()

    chunks = []
    with args.chunks.open(encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    if args.max_chunks:
        chunks = chunks[: args.max_chunks]

    print(f"Чанков для обработки: {len(chunks)}")
    print(f"Бэкенд: {args.backend} / {args.model}")

    call_fn = call_openai if args.backend == "openai" else call_ollama

    total_queries = 0
    skipped = 0

    with args.output.open("w", encoding="utf-8") as fout:
        for chunk in tqdm(chunks, desc="Генерация вопросов"):
            try:
                queries = call_fn(
                    text=chunk["text"],
                    subdomain=chunk["metadata"].get("subdomain", "общая_геология"),
                    n=args.queries_per_chunk,
                    model=args.model,
                )
                queries = [q for q in queries if filter_query(q)]

                if not queries:
                    skipped += 1
                    continue

                for query in queries:
                    record = {
                        "query": query,
                        "positive_chunk_id": chunk["id"],
                        "positive_text": chunk["text"],
                        "subdomain": chunk["metadata"].get("subdomain"),
                        "source": chunk["metadata"].get("source"),
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_queries += 1

                time.sleep(args.sleep + random.uniform(0, 0.3))

            except Exception as e:
                print(f"\n[error] chunk {chunk['id']}: {e}")
                skipped += 1

    print(f"\n✓ Сгенерировано вопросов: {total_queries}")
    print(f"  Пропущено чанков: {skipped}")
    print(f"  Среднее вопросов/чанк: {total_queries / max(len(chunks) - skipped, 1):.1f}")


if __name__ == "__main__":
    main()
