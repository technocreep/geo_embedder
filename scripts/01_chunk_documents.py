"""
01_chunk_documents.py
=====================
Чанкование геологических документов с извлечением доменных метаданных.

Поддерживаемые форматы: PDF, TXT, DOCX, MD
Выход: JSONL файл с чанками + метаданными

Запуск:
    python 01_chunk_documents.py --input_dir ./raw_docs --output ./chunks.jsonl
"""

import argparse
import asyncio
import json
import hashlib
import os
from pathlib import Path
from typing import Optional

# pip install pypdf python-docx langchain-text-splitters tqdm openai
from pypdf import PdfReader
import docx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
from openai import AsyncOpenAI

from dotenv import load_dotenv

load_dotenv()


# ─── Геологические подобласти ────────────────────────────────────────────────
GEO_SUBDOMAINS = [
    "стратиграфия",
    "петрография",
    "тектоника",
    "геохимия",
    "гидрогеология",
    "рудная_геология",
    "сейсмика",
    "гис_картография",
    "геофизика",
    "общая_геология",
]

_SUBDOMAIN_SET = set(GEO_SUBDOMAINS)
_DOMAINS_LIST = ", ".join(GEO_SUBDOMAINS)

_SYSTEM_PROMPT = (
    f"Ты классификатор геологических текстов. "
    f"Выбери одну подобласть из списка: {_DOMAINS_LIST}. "
    f"Отвечай строго одним словом — только названием подобласти, без пояснений."
)


# ─── Асинхронный LLM-клиент ──────────────────────────────────────────────────

def _make_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url=os.environ.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1"),
    )


async def detect_subdomain_async(
    text: str,
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    model: str,
    pbar: tqdm,
) -> str:
    """Определяет подобласть геологии с помощью LLM (асинхронно)."""
    async with sem:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": text[:500]},
                ],
                max_tokens=12,
                temperature=0,
            )
            result = response.choices[0].message.content.strip().lower().replace(" ", "_")
            if result in _SUBDOMAIN_SET:
                return result
            for domain in GEO_SUBDOMAINS:
                if domain in result:
                    return domain
            return "общая_геология"
        except Exception as e:
            tqdm.write(f"  [llm error] detect_subdomain: {e}")
            return "общая_геология"
        finally:
            pbar.update(1)


# ─── Извлечение текста ────────────────────────────────────────────────────────

def extract_text_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


def extract_text_docx(path: Path) -> str:
    doc = docx.Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def extract_text(path: Path) -> Optional[str]:
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            return extract_text_pdf(path)
        elif suffix == ".docx":
            return extract_text_docx(path)
        elif suffix in (".txt", ".md"):
            return path.read_text(encoding="utf-8", errors="ignore")
        else:
            tqdm.write(f"  [skip] Unsupported format: {path}")
            return None
    except Exception as e:
        tqdm.write(f"  [error] {path.name}: {e}")
        return None


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """
    Рекурсивное разбиение с учётом геологических разделителей.
    Порядок разделителей настроен под структуру геол. отчётов.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=[
            "\n\n\n",        # Разделы отчёта
            "\n\n",          # Абзацы
            "\n",            # Строки
            ". ",            # Предложения
            ", ",
            " ",
            "",
        ],
        length_function=len,
    )
    return splitter.split_text(text)


def make_chunk_id(source: str, chunk_index: int, text: str) -> str:
    h = hashlib.md5(f"{source}_{chunk_index}_{text[:50]}".encode()).hexdigest()[:8]
    return f"{Path(source).stem}_{chunk_index:04d}_{h}"


# ─── Основной пайплайн ────────────────────────────────────────────────────────

async def process_directory(
    input_dir: Path,
    output_path: Path,
    chunk_size: int = 512,
    overlap: int = 64,
    concurrency: int = 10,
):
    files = list(input_dir.rglob("*"))
    files = [f for f in files if f.suffix.lower() in (".pdf", ".docx", ".txt", ".md")]
    print(f"Найдено файлов: {len(files)}")

    # ── Фаза 1: извлечение текста и чанкование (CPU, без LLM) ────────────────
    print("\nФаза 1/2 — извлечение и чанкование документов")
    raw_chunks: list[dict] = []  # {id, text, metadata} без subdomain
    skipped_files = 0

    for fpath in tqdm(files, desc="  Документы", unit="файл"):
        text = extract_text(fpath)
        if not text or len(text.strip()) < 100:
            skipped_files += 1
            continue

        chunks = chunk_text(text, chunk_size, overlap)
        total_in_doc = len(chunks)

        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:
                continue
            raw_chunks.append({
                "id": make_chunk_id(str(fpath), i, chunk),
                "text": chunk.strip(),
                "metadata": {
                    "source": fpath.name,
                    "source_path": str(fpath.relative_to(input_dir)),
                    "chunk_index": i,
                    "total_chunks": total_in_doc,
                    "char_length": len(chunk),
                },
            })

    print(f"  Извлечено чанков: {len(raw_chunks):,}  |  пропущено файлов: {skipped_files}")

    # ── Фаза 2: асинхронное определение подобластей (LLM) ────────────────────
    model = os.environ["DOMAIN_DETECTION_MODEL"]
    print(f"\nФаза 2/2 — определение подобластей  [{model}, параллелизм={concurrency}]")

    client = _make_client()
    sem = asyncio.Semaphore(concurrency)

    with tqdm(total=len(raw_chunks), desc="  Чанки", unit="чанк") as pbar:
        tasks = [
            detect_subdomain_async(c["text"], client, sem, model, pbar)
            for c in raw_chunks
        ]
        subdomains = await asyncio.gather(*tasks)

    await client.close()

    # ── Запись результатов ────────────────────────────────────────────────────
    subdomain_counts: dict[str, int] = {}
    with output_path.open("w", encoding="utf-8") as fout:
        for chunk, subdomain in zip(raw_chunks, subdomains):
            chunk["metadata"]["subdomain"] = subdomain
            subdomain_counts[subdomain] = subdomain_counts.get(subdomain, 0) + 1
            fout.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    total_chunks = len(raw_chunks)
    print(f"\n✓ Сохранено чанков: {total_chunks:,}  →  {output_path}")
    print(f"\nРаспределение по подобластям:")
    for domain, cnt in sorted(subdomain_counts.items(), key=lambda x: -x[1]):
        pct = cnt / total_chunks * 100
        print(f"  {domain:30s} {cnt:5d} ({pct:.1f}%)")


def main():
    print("Чанкование геологических документов")
    parser = argparse.ArgumentParser(description="Чанкование геологических документов")
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("chunks.jsonl"))
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Число параллельных LLM-запросов (по умолчанию 10)")
    args = parser.parse_args()

    asyncio.run(process_directory(
        args.input_dir, args.output, args.chunk_size, args.overlap, args.concurrency,
    ))


if __name__ == "__main__":
    main()
