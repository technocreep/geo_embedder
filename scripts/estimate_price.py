"""
estimate_price.py
=================
Предварительная оценка стоимости LLM-вызовов пайплайна.

Пайплайн (только шаги с LLM):
  Шаг 1  01_chunk_documents.py   — все чанки  → DOMAIN_DETECTION_MODEL
  Шаг 2  02_generate_queries.py  — все чанки  → QUERY_MODEL
  [Шаг 3  00_split_data.py       — разделение на train/test, LLM не используется]
  Шаг 4  03_mine_hard_negatives  — train-чанки × adversarial_ratio → ADVERSARIAL_MODEL

Читает модели из переменных окружения (.env), цены из utils/llm_price.json (за 1M токенов).

Запуск:
    python scripts/estimate_price.py --chunks all_chunks.jsonl
    python scripts/estimate_price.py --num_chunks 1000
    python scripts/estimate_price.py --num_chunks 1000 --test_ratio 0.2 --queries_per_chunk 3 --adversarial_ratio 0.3
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

# ─── Накладные расходы промптов (токены, из фактических промптов в скриптах) ──

# Шаг 1: системный промпт классификатора + список доменов
S1_SYSTEM_TOKENS    = 90
S1_MAX_USER_CHARS   = 500   # text[:500] в detect_subdomain_async
S1_OUTPUT_TOKENS    = 4     # одно слово-домен

# Шаг 2: системный промпт эксперта + шаблон пользовательского запроса
S2_SYSTEM_TOKENS    = 200
S2_TEMPLATE_TOKENS  = 65    # USER_PROMPT_TEMPLATE без текста чанка
S2_MAX_USER_CHARS   = 2000  # text[:2000] в call_openai
S2_TOKENS_PER_QUERY = 50    # ~50 токенов на один вопрос

# Шаг 4: adversarial-промпт без текста
S4_PROMPT_TOKENS    = 145
S4_MAX_USER_CHARS   = 1500  # text[:1500] в build_adversarial_negatives
S4_OUTPUT_RATIO     = 0.9   # перефразировка ≈ той же длины


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
        print(f"[warn] {env_var} не задана в .env")
    return val


def cost_usd(input_tokens: int, output_tokens: int, model: str, prices: dict) -> float | None:
    """Стоимость в USD. Цены в llm_price.json указаны за 1 млн токенов."""
    if model not in prices:
        return None
    p = prices[model]
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000


def get_chunk_stats(chunks_path: Path) -> tuple[int, float]:
    """Возвращает (кол-во чанков, среднюю длину текста в символах)."""
    lengths = []
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            lengths.append(len(obj.get("text", "")))
    if not lengths:
        return 0, 0.0
    return len(lengths), sum(lengths) / len(lengths)


# ─── Оценки по шагам ──────────────────────────────────────────────────────────

def estimate_step1(n_all: int, avg_chars: float, model: str, prices: dict) -> dict:
    """Шаг 1: detect_subdomain для каждого чанка."""
    user_chars = min(avg_chars, S1_MAX_USER_CHARS)
    in_tok  = S1_SYSTEM_TOKENS + chars_to_tokens(int(user_chars))
    out_tok = S1_OUTPUT_TOKENS
    total_in  = in_tok  * n_all
    total_out = out_tok * n_all
    return {
        "step":   "Шаг 1 — 01_chunk_documents.py  (detect_subdomain)",
        "scope":  f"все чанки ({n_all:,})",
        "model":  model,
        "calls":  n_all,
        "input_tokens":  total_in,
        "output_tokens": total_out,
        "usd":    cost_usd(total_in, total_out, model, prices),
    }


def estimate_step2(
    n_all: int,
    avg_chars: float,
    model: str,
    prices: dict,
    queries_per_chunk: int,
) -> dict:
    """Шаг 2: generate_queries для каждого чанка."""
    user_chars = min(avg_chars, S2_MAX_USER_CHARS)
    in_tok  = S2_SYSTEM_TOKENS + S2_TEMPLATE_TOKENS + chars_to_tokens(int(user_chars))
    out_tok = S2_TOKENS_PER_QUERY * queries_per_chunk
    total_in  = in_tok  * n_all
    total_out = out_tok * n_all
    return {
        "step":   "Шаг 2 — 02_generate_queries.py",
        "scope":  f"все чанки ({n_all:,}), {queries_per_chunk} вопр./чанк",
        "model":  model,
        "calls":  n_all,
        "input_tokens":  total_in,
        "output_tokens": total_out,
        "usd":    cost_usd(total_in, total_out, model, prices),
    }


def estimate_step4(
    n_all: int,
    avg_chars: float,
    model: str,
    prices: dict,
    test_ratio: float,
    adversarial_ratio: float,
) -> dict:
    """Шаг 4: adversarial negatives — только train-чанки × adversarial_ratio."""
    n_train = round(n_all * (1.0 - test_ratio))
    n_adv   = round(n_train * adversarial_ratio)
    user_chars = min(avg_chars, S4_MAX_USER_CHARS)
    in_tok  = S4_PROMPT_TOKENS + chars_to_tokens(int(user_chars))
    out_tok = round(chars_to_tokens(int(user_chars)) * S4_OUTPUT_RATIO)
    total_in  = in_tok  * n_adv
    total_out = out_tok * n_adv
    return {
        "step":   "Шаг 4 — 03_mine_hard_negatives.py  (adversarial)",
        "scope":  f"train-чанки ({n_train:,}) × {adversarial_ratio} = {n_adv:,}",
        "model":  model,
        "calls":  n_adv,
        "input_tokens":  total_in,
        "output_tokens": total_out,
        "usd":    cost_usd(total_in, total_out, model, prices),
    }


# ─── Вывод ────────────────────────────────────────────────────────────────────

def print_report(
    estimates: list[dict],
    prices: dict,
    n_all: int,
    n_train: int,
    n_test: int,
) -> None:
    print()
    print("=" * 100)
    print("  ОЦЕНКА СТОИМОСТИ ПАЙПЛАЙНА  (цены из llm_price.json, за 1M токенов)")
    print("=" * 100)
    print(f"  Всего чанков: {n_all:,}  |  train: {n_train:,}  |  test: {n_test:,}")
    print("-" * 100)

    step_w  = 52
    scope_w = 34
    print(
        f"  {'Этап':<{step_w}} {'Охват':<{scope_w}} {'Модель':<36}"
        f" {'In, Ktok':>9} {'Out, Ktok':>10} {'USD':>9}"
    )
    print("-" * 100)

    total_usd  = 0.0
    has_unknown = False

    for e in estimates:
        model_label = e["model"] or "[не задана]"
        if e["usd"] is None:
            usd_str = "нет цены"
            has_unknown = True
        else:
            usd_str = f"${e['usd']:.4f}"
            total_usd += e["usd"]

        print(
            f"  {e['step']:<{step_w}} "
            f"{e['scope']:<{scope_w}} "
            f"{model_label:<36} "
            f"{e['input_tokens'] / 1000:>9.1f} "
            f"{e['output_tokens'] / 1000:>10.1f} "
            f"{usd_str:>9}"
        )

    print("-" * 100)
    total_str = f"${total_usd:.4f}" + ("  (+ шаги без цены)" if has_unknown else "")
    print(f"  {'ИТОГО':>{step_w + scope_w + 37}} {total_str:>20}")
    print("=" * 100)

    missing = [
        m for m in {e["model"] for e in estimates if e["model"]}
        if m not in prices
    ]
    if missing:
        print(f"\n  [!] Модели отсутствуют в {PRICE_FILE.name}: {', '.join(missing)}")
        print(f"      Добавьте их в utils/llm_price.json для полного расчёта.")
    print()


def main():
    parser = argparse.ArgumentParser(description="Оценка стоимости LLM-вызовов пайплайна geo-embedder")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--chunks", type=Path,
                     help="Путь к all_chunks.jsonl (точная оценка по реальным данным)")
    src.add_argument("--num_chunks", type=int,
                     help="Количество чанков (если файл ещё не создан)")
    parser.add_argument("--avg_chars", type=int, default=400,
                        help="Средняя длина чанка в символах при --num_chunks (по умолчанию 400)")
    parser.add_argument("--test_ratio", type=float, default=0.2,
                        help="Доля test-выборки, как в 00_split_data.py (по умолчанию 0.2)")
    parser.add_argument("--queries_per_chunk", type=int, default=3,
                        help="Вопросов на чанк для шага 2 (по умолчанию 3)")
    parser.add_argument("--adversarial_ratio", type=float, default=0.3,
                        help="Доля train-чанков для adversarial в шаге 4 (по умолчанию 0.3)")
    args = parser.parse_args()

    prices = load_prices()

    if args.chunks:
        if not args.chunks.exists():
            print(f"[error] Файл не найден: {args.chunks}")
            return
        n_all, avg_chars = get_chunk_stats(args.chunks)
        print(f"Чанков в файле: {n_all:,}, средняя длина: {avg_chars:.0f} символов")
    else:
        n_all     = args.num_chunks
        avg_chars = float(args.avg_chars)
        print(f"Чанков (оценка): {n_all:,}, средняя длина: {avg_chars:.0f} символов")

    n_train = round(n_all * (1.0 - args.test_ratio))
    n_test  = n_all - n_train

    model_01 = get_model("DOMAIN_DETECTION_MODEL")
    model_02 = get_model("QUERY_MODEL")
    model_03 = get_model("ADVERSARIAL_MODEL")

    estimates = [
        estimate_step1(n_all, avg_chars, model_01, prices),
        estimate_step2(n_all, avg_chars, model_02, prices, args.queries_per_chunk),
        estimate_step4(n_all, avg_chars, model_03, prices, args.test_ratio, args.adversarial_ratio),
    ]

    print_report(estimates, prices, n_all, n_train, n_test)


if __name__ == "__main__":
    main()
