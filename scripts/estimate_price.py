"""
estimate_price.py
=================
Предварительная оценка стоимости запуска пайплайна (скрипты 01, 02, 03).

Читает модели из переменных окружения, цены из utils/llm_price.json.
Для точных оценок передайте файл чанков; без него — укажите --num_chunks.

Запуск:
    python scripts/estimate_price.py --chunks chunks.jsonl
    python scripts/estimate_price.py --num_chunks 1000
    python scripts/estimate_price.py --num_chunks 1000 --queries_per_chunk 5 --adversarial_ratio 0.5
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ─── Пути ─────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.parent
PRICE_FILE = REPO_ROOT / "utils" / "llm_price.json"

# ─── Параметры аппроксимации токенов ─────────────────────────────────────────
# Для русскоязычного текста большинство токенизаторов дают ~3 символа/токен.
CHARS_PER_TOKEN = 3.0

# ─── Фиксированные накладные расходы промптов (в токенах) ────────────────────
# Значения получены из фактических промптов в скриптах.

# 01: системный промпт классификатора + список доменов
SCRIPT01_SYSTEM_TOKENS = 90
# 01: пользовательский контент ограничен 500 символами
SCRIPT01_MAX_USER_CHARS = 500
# 01: ответ — одно слово-домен
SCRIPT01_OUTPUT_TOKENS = 4

# 02: системный промпт эксперта-геолога
SCRIPT02_SYSTEM_TOKENS = 200
# 02: шаблон пользовательского промпта без текста чанка
SCRIPT02_USER_TEMPLATE_TOKENS = 65
# 02: текст чанка ограничен 2000 символами
SCRIPT02_MAX_USER_CHARS = 2000
# 02: ответ — JSON-массив вопросов (~50 токенов на вопрос)
SCRIPT02_TOKENS_PER_QUERY = 50

# 03: промпт adversarial-перефразировки без текста
SCRIPT03_PROMPT_TOKENS = 145
# 03: текст чанка ограничен 1500 символами
SCRIPT03_MAX_USER_CHARS = 1500
# 03: ответ — перефразированный текст сопоставимой длины (~0.9 от входного)
SCRIPT03_OUTPUT_RATIO = 0.9


def chars_to_tokens(chars: int) -> int:
    return max(1, round(chars / CHARS_PER_TOKEN))


def load_prices() -> dict:
    if not PRICE_FILE.exists():
        print(f"[warn] Файл цен не найден: {PRICE_FILE}")
        return {}
    with PRICE_FILE.open(encoding="utf-8") as f:
        return json.load(f)


def get_model(env_var: str) -> str:
    val = os.environ.get(env_var, "")
    if not val:
        print(f"[warn] Переменная окружения {env_var} не задана")
    return val


def cost_usd(input_tokens: int, output_tokens: int, model: str, prices: dict) -> float | None:
    """Возвращает стоимость в USD или None если модель не найдена в прайсе."""
    if model not in prices:
        return None
    p = prices[model]
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000


def get_chunk_stats(chunks_path: Path) -> tuple[int, float]:
    """Возвращает (количество чанков, среднюю длину текста в символах)."""
    lengths = []
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            lengths.append(len(obj.get("text", "")))
    if not lengths:
        return 0, 0.0
    return len(lengths), sum(lengths) / len(lengths)


def estimate_script01(num_chunks: int, avg_chars: float, model: str, prices: dict) -> dict:
    user_chars = min(avg_chars, SCRIPT01_MAX_USER_CHARS)
    input_tok = SCRIPT01_SYSTEM_TOKENS + chars_to_tokens(int(user_chars))
    output_tok = SCRIPT01_OUTPUT_TOKENS

    total_in = input_tok * num_chunks
    total_out = output_tok * num_chunks
    usd = cost_usd(total_in, total_out, model, prices)

    return {
        "script": "01_chunk_documents.py  (detect_subdomain)",
        "model": model,
        "calls": num_chunks,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "usd": usd,
    }


def estimate_script02(
    num_chunks: int,
    avg_chars: float,
    model: str,
    prices: dict,
    queries_per_chunk: int,
) -> dict:
    user_chars = min(avg_chars, SCRIPT02_MAX_USER_CHARS)
    input_tok = SCRIPT02_SYSTEM_TOKENS + SCRIPT02_USER_TEMPLATE_TOKENS + chars_to_tokens(int(user_chars))
    output_tok = SCRIPT02_TOKENS_PER_QUERY * queries_per_chunk

    total_in = input_tok * num_chunks
    total_out = output_tok * num_chunks
    usd = cost_usd(total_in, total_out, model, prices)

    return {
        "script": "02_generate_queries.py",
        "model": model,
        "calls": num_chunks,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "usd": usd,
    }


def estimate_script03(
    num_chunks: int,
    avg_chars: float,
    model: str,
    prices: dict,
    adversarial_ratio: float,
) -> dict:
    adv_chunks = round(num_chunks * adversarial_ratio)
    user_chars = min(avg_chars, SCRIPT03_MAX_USER_CHARS)
    input_tok = SCRIPT03_PROMPT_TOKENS + chars_to_tokens(int(user_chars))
    output_tok = round(chars_to_tokens(int(user_chars)) * SCRIPT03_OUTPUT_RATIO)

    total_in = input_tok * adv_chunks
    total_out = output_tok * adv_chunks
    usd = cost_usd(total_in, total_out, model, prices)

    return {
        "script": f"03_mine_hard_negatives.py  (adversarial, ratio={adversarial_ratio})",
        "model": model,
        "calls": adv_chunks,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "usd": usd,
    }


def print_report(estimates: list[dict], prices: dict) -> None:
    col_w = 55
    print()
    print("=" * 90)
    print("  ОЦЕНКА СТОИМОСТИ ПАЙПЛАЙНА")
    print("=" * 90)
    print(f"  {'Скрипт':<{col_w}} {'Модель':<35} {'Вызовов':>8} {'In, Ktok':>10} {'Out, Ktok':>10} {'USD':>10}")
    print("-" * 90)

    total_usd = 0.0
    has_unknown = False

    for e in estimates:
        model_label = e["model"] if e["model"] else "[не задана]"
        usd_str = f"${e['usd']:.3f}" if e["usd"] is not None else "нет цены"
        if e["usd"] is None:
            has_unknown = True
        else:
            total_usd += e["usd"]

        print(
            f"  {e['script']:<{col_w}} "
            f"{model_label:<35} "
            f"{e['calls']:>8,} "
            f"{e['input_tokens']/1000:>10.1f} "
            f"{e['output_tokens']/1000:>10.1f} "
            f"{usd_str:>10}"
        )

    print("-" * 90)
    total_str = f"${total_usd:.3f}" + (" (+ неизвестные)" if has_unknown else "")
    print(f"  {'ИТОГО':<{col_w + 35 + 9}} {total_str:>31}")
    print("=" * 90)

    missing = [m for m in set(e["model"] for e in estimates if e["model"]) if m not in prices]
    if missing:
        print(f"\n  [!] Модели не найдены в {PRICE_FILE.name}: {', '.join(missing)}")
        print(f"      Добавьте их в utils/llm_price.json для полного расчёта.")
    print()


def main():
    parser = argparse.ArgumentParser(description="Оценка стоимости пайплайна geo-embedder")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--chunks", type=Path, help="Путь к chunks.jsonl для точной оценки")
    src.add_argument("--num_chunks", type=int, help="Количество чанков (если файл недоступен)")
    parser.add_argument("--avg_chars", type=int, default=400,
                        help="Средняя длина чанка в символах (при --num_chunks, по умолчанию 400)")
    parser.add_argument("--queries_per_chunk", type=int, default=3,
                        help="Вопросов на чанк для скрипта 02 (по умолчанию 3)")
    parser.add_argument("--adversarial_ratio", type=float, default=0.3,
                        help="Доля чанков для adversarial в скрипте 03 (по умолчанию 0.3)")
    args = parser.parse_args()

    prices = load_prices()

    if args.chunks:
        if not args.chunks.exists():
            print(f"[error] Файл не найден: {args.chunks}")
            return
        num_chunks, avg_chars = get_chunk_stats(args.chunks)
        print(f"Чанков в файле: {num_chunks:,}, средняя длина: {avg_chars:.0f} символов")
    else:
        num_chunks = args.num_chunks
        avg_chars = args.avg_chars
        print(f"Чанков (оценка): {num_chunks:,}, средняя длина: {avg_chars} символов")

    model_01 = get_model("DOMAIN_DETECTION_MODEL")
    model_02 = get_model("QUERY_MODEL")
    model_03 = get_model("ADVERSARIAL_MODEL")

    estimates = [
        estimate_script01(num_chunks, avg_chars, model_01, prices),
        estimate_script02(num_chunks, avg_chars, model_02, prices, args.queries_per_chunk),
        estimate_script03(num_chunks, avg_chars, model_03, prices, args.adversarial_ratio),
    ]

    print_report(estimates, prices)


if __name__ == "__main__":
    main()
