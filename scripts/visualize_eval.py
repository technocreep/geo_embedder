"""
visualize_eval.py
=================
Визуализация результатов сравнительного бенчмарка моделей (eval_results.json).

Запуск:
    python scripts/visualize_eval.py --results eval_results.json --output output/eval_plots.png

Генерирует отдельные PNG для каждого графика и CSV с таблицей метрик.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


METRICS = ["NDCG@5", "NDCG@10", "Recall@5", "Recall@10", "MRR"]
METRIC_LABELS = ["NDCG@5", "NDCG@10", "Recall@5", "Recall@10", "MRR"]

# Короткие имена для осей
SHORT_NAMES = {
    "/models/finetuned-bge-m3-geo":     "BGE-M3\nFine-tuned",
    "BAAI/bge-m3":                       "BGE-M3\nBaseline",
    "google/embeddinggemma-300m":        "Gemma\n300M",
    "yasserrmd/geo-gemma-300m-emb":      "geo-Gemma\n300M",
    "ai-forever/sbert_large_nlu_ru":     "SBERT\nru-large",
    "intfloat/multilingual-e5-large":    "mE5\nlarge",
    "voyageai/voyage-4-nano":    "voyage\n4-nano",
}

COLORS = {
    "/models/finetuned-bge-m3-geo":  "#16a34a",   # зелёный — наш fine-tuned
    "BAAI/bge-m3":                    "#2563eb",   # синий — baseline
    "google/embeddinggemma-300m":     "#9333ea",   # фиолетовый
    "yasserrmd/geo-gemma-300m-emb":   "#ea580c",   # оранжевый
    "ai-forever/sbert_large_nlu_ru":  "#f322c6",   # красный
    "intfloat/multilingual-e5-large": "#0891b2",   # голубой
    "voyageai/voyage-4-nano": "#d5dd3f",
}


def load_results(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _subtitle(results: list[dict]) -> str:
    return (f"{results[0]['n_queries']:,} запросов, {results[0]['n_corpus']:,} чанков")


def plot_all_metrics(results: list[dict], output_path: Path):
    """Grouped bar chart: все метрики для всех моделей."""
    models = [r["model"] for r in results]
    short  = [SHORT_NAMES.get(m, m.split("/")[-1]) for m in models]
    colors = [COLORS.get(m, "#6b7280") for m in models]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(f"Все метрики\n({_subtitle(results)})", fontsize=13, fontweight="bold")

    x = np.arange(len(METRICS))
    n_models = len(models)
    width = 0.12
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    for i, (r, color, sname) in enumerate(zip(results, colors, short)):
        vals = [r[m] for m in METRICS]
        bars = ax.bar(x + offsets[i], vals, width, color=color, alpha=0.85,
                      label=sname.replace("\n", " "))
        if r["model"] in ("/models/finetuned-bge-m3-geo", "BAAI/bge-m3"):
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=6, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS, fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ График сохранён: {output_path}")


def plot_ndcg10(results: list[dict], output_path: Path):
    """Horizontal bar: NDCG@10 (основная метрика) с дельтой к baseline."""
    models = [r["model"] for r in results]
    short  = [SHORT_NAMES.get(m, m.split("/")[-1]) for m in models]
    colors = [COLORS.get(m, "#6b7280") for m in models]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(f"NDCG@10 (основная метрика)\n({_subtitle(results)})", fontsize=13, fontweight="bold")

    ndcg10s = [r["NDCG@10"] for r in results]
    y_pos = np.arange(len(models))

    bars = ax.barh(y_pos, ndcg10s, color=colors, alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, ndcg10s):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=9, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(short, fontsize=8)
    ax.set_xlabel("NDCG@10")
    ax.set_xlim(0, 1.05)
    ax.axvline(ndcg10s[0], color=colors[0], linestyle="--", linewidth=1, alpha=0.5)
    ax.grid(axis="x", alpha=0.3)

    baseline_ndcg = next(r["NDCG@10"] for r in results if r["model"] == "BAAI/bge-m3")
    # for i, (v, model) in enumerate(zip(ndcg10s, models)):
    #     if model != "BAAI/bge-m3":
    #         delta = v - baseline_ndcg
    #         sign = "+" if delta >= 0 else ""
    #         color = "#16a34a" if delta > 0 else "#dc2626"
    #         ax.text(0.01, i, f"  {sign}{delta:+.3f} vs baseline",
    #                 va="center", fontsize=7, color=color, style="italic")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ График сохранён: {output_path}")


def plot_scatter(results: list[dict], output_path: Path):
    """Scatter: Recall@10 vs NDCG@10 (качество vs полнота)."""
    models = [r["model"] for r in results]
    short  = [SHORT_NAMES.get(m, m.split("/")[-1]) for m in models]
    colors = [COLORS.get(m, "#6b7280") for m in models]

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(f"Качество ранжирования vs Полнота\n({_subtitle(results)})", fontsize=13, fontweight="bold")

    for r, color, sname in zip(results, colors, short):
        ax.scatter(r["Recall@10"], r["NDCG@10"], s=120, color=color,
                   zorder=5, edgecolors="white", linewidth=0.8)
        ax.annotate(sname.replace("\n", " "), (r["Recall@10"], r["NDCG@10"]),
                    textcoords="offset points", xytext=(6, 3), fontsize=7)

    ax.set_xlabel("Recall@10")
    ax.set_ylabel("NDCG@10")
    ax.grid(alpha=0.3)

    patches = [mpatches.Patch(color=c, label=s.replace("\n", " "))
               for c, s in zip(colors, short)]
    ax.legend(handles=patches, fontsize=7, loc="lower right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ График сохранён: {output_path}")


def save_csv(results: list[dict], output_path: Path):
    """Сохранить таблицу метрик в CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "short_name"] + METRICS + ["time_sec", "n_queries", "n_corpus"]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            short_name = SHORT_NAMES.get(r["model"], r["model"].split("/")[-1]).replace("\n", " ")
            row = {
                "model": r["model"],
                "short_name": short_name,
                **{m: f"{r[m]:.4f}" for m in METRICS},
                "time_sec": f"{r['time_sec']:.0f}",
                "n_queries": r["n_queries"],
                "n_corpus": r["n_corpus"],
            }
            writer.writerow(row)

    print(f"✓ Таблица сохранена: {output_path}")


def print_summary(results: list[dict]):
    print("\n── Результаты бенчмарка ───────────────────────────────────────")
    print(f"{'Модель':35s} {'NDCG@10':>8} {'Recall@10':>10} {'MRR':>8} {'time':>6}")
    print("─" * 72)
    for r in results:
        name = SHORT_NAMES.get(r["model"], r["model"]).replace("\n", " ")
        marker = " ←" if r["model"] == "/models/finetuned-bge-m3-geo" else ""
        print(f"{name:35s} {r['NDCG@10']:>8.4f} {r['Recall@10']:>10.4f} "
              f"{r['MRR']:>8.4f} {r['time_sec']:>5.0f}s{marker}")
    print("─" * 72)

    ft_list = [r for r in results if r["model"] == "/models/finetuned-bge-m3-geo"]
    bl_list = [r for r in results if r["model"] == "BAAI/bge-m3"]
    if ft_list and bl_list:
        ft, bl = ft_list[0], bl_list[0]
        print(f"\nДельта fine-tuned vs baseline:")
        for m in METRICS:
            d = ft[m] - bl[m]
            print(f"  {m:12s}: {d:+.4f} ({d/bl[m]*100:+.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("eval_results.json"))
    parser.add_argument("--output", type=Path, default=Path("output/eval_plots.png"),
                        help="Базовый путь; имя файла используется как префикс для отдельных изображений")
    args = parser.parse_args()

    results = load_results(args.results)
    print(f"Моделей в бенчмарке: {len(results)}")

    stem = args.output.stem
    out_dir = args.output.parent

    plot_all_metrics(results, out_dir / f"{stem}_all_metrics.png")
    plot_ndcg10(results,      out_dir / f"{stem}_ndcg10.png")
    plot_scatter(results,     out_dir / f"{stem}_scatter.png")
    save_csv(results,         out_dir / f"{stem}_table.csv")
    print_summary(results)


if __name__ == "__main__":
    main()
