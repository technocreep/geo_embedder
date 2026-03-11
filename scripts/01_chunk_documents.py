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
import json
import re
import hashlib
from pathlib import Path
from typing import Optional

# pip install pypdf python-docx langchain-text-splitters tqdm
from pypdf import PdfReader
import docx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm


# ─── Геологические подобласти (для subdomain labeling) ───────────────────────
GEO_SUBDOMAINS = {
    "стратиграфия": ["стратиграф", "горизонт", "свита", "ярус", "эпоха", "период",
                     "кайнозой", "мезозой", "палеозой", "протерозой"],
    "петрография": ["порода", "магматич", "осадоч", "метаморф", "минерал", "кристалл",
                    "гранит", "известняк", "сланец", "песчаник"],
    "тектоника": ["разлом", "сброс", "надвиг", "складч", "деформац", "тектоник",
                  "плита", "блок", "грабен", "горст"],
    "геохимия": ["геохими", "элемент", "изотоп", "концентрац", "аномали",
                 "халькофил", "литофил", "сидерофил"],
    "гидрогеология": ["водонос", "подземн вод", "фильтрац", "водопроницаем",
                      "дебит", "пьезометр", "водоупор"],
    "нефтегазовая": ["нефт", "газ", "углеводород", "коллектор", "пластов",
                     "керн", "скважин", "пористост", "проницаем"],
    "рудная_геология": ["руда", "рудн", "место-рожден", "месторожден",
                        "золото", "медь", "полиметалл", "прогнозн"],
    "сейсмика": ["сейсм", "отражени", "преломлени", "волна", "горизонт сейсм",
                 "инверси", "атрибут"],
    "гис_картография": ["ГИС", "картографи", "геодез", "координат", "проекц",
                        "топограф", "дистанционн"],
    "геофизика": ["геофизик", "гравиметр", "магнитометр", "электроразведк",
                  "каротаж", "ВЭЗ", "МТЗ"],
}


def detect_subdomain(text: str) -> str:
    """Определяет подобласть геологии по ключевым словам."""
    text_lower = text.lower()
    scores = {}
    for domain, keywords in GEO_SUBDOMAINS.items():
        score = sum(1 for kw in keywords if kw.lower() in text_lower)
        if score > 0:
            scores[domain] = score
    if not scores:
        return "общая_геология"
    return max(scores, key=scores.get)


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
            print(f"  [skip] Unsupported format: {path}")
            return None
    except Exception as e:
        print(f"  [error] {path.name}: {e}")
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


def process_directory(
    input_dir: Path,
    output_path: Path,
    chunk_size: int = 512,
    overlap: int = 64,
):
    files = list(input_dir.rglob("*"))
    files = [f for f in files if f.suffix.lower() in (".pdf", ".docx", ".txt", ".md")]
    print(f"Найдено файлов: {len(files)}")

    total_chunks = 0
    subdomain_counts: dict[str, int] = {}

    with output_path.open("w", encoding="utf-8") as fout:
        for fpath in tqdm(files, desc="Обработка документов"):
            text = extract_text(fpath)
            if not text or len(text.strip()) < 100:
                continue

            chunks = chunk_text(text, chunk_size, overlap)

            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) < 50:   # Пропускаем слишком короткие
                    continue

                subdomain = detect_subdomain(chunk)
                subdomain_counts[subdomain] = subdomain_counts.get(subdomain, 0) + 1

                record = {
                    "id": make_chunk_id(str(fpath), i, chunk),
                    "text": chunk.strip(),
                    "metadata": {
                        "source": fpath.name,
                        "source_path": str(fpath.relative_to(input_dir)),
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "subdomain": subdomain,
                        "char_length": len(chunk),
                    }
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_chunks += 1

    print(f"\n✓ Всего чанков: {total_chunks}")
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
    args = parser.parse_args()

    process_directory(args.input_dir, args.output, args.chunk_size, args.overlap)


if __name__ == "__main__":
    main()
