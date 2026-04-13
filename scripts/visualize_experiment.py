"""
visualize_experiment.py
=======================
Парсит лог обучения (experiment.txt) и строит графики:
  - Train loss по эпохам
  - Eval метрики (NDCG@10, Accuracy@1, MRR@10) по эпохам
  - Learning rate schedule
  - Grad norm по эпохам

Запуск:
    python scripts/visualize_experiment.py --log experiment.txt --output output/training_plots.png
"""

import argparse
import ast
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def parse_log(path: Path):
    loss_data = []       # (epoch, loss, lr, grad_norm)
    eval_data = []       # (epoch, ndcg10, acc1, mrr10, recall10, map100)

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Строки с loss: {'loss': '1.234', 'grad_norm': '...', 'learning_rate': '...', 'epoch': '...'}
            if line.startswith("{'loss':"):
                try:
                    d = ast.literal_eval(line)
                    loss_data.append({
                        "epoch": float(d["epoch"]),
                        "loss": float(d["loss"]),
                        "lr": float(d.get("learning_rate", 0)),
                        "grad_norm": float(d.get("grad_norm", 0)),
                    })
                except Exception:
                    pass

            # Строки с eval: {'eval_geo-dev_cos_sim_ndcg@10': '...', ..., 'epoch': '...'}
            if line.startswith("{'eval_geo-dev"):
                try:
                    d = ast.literal_eval(line)
                    epoch = float(d.get("epoch", 0))
                    ndcg10 = float(d.get("eval_geo-dev_cos_sim_ndcg@10", 0))
                    acc1 = float(d.get("eval_geo-dev_cos_sim_accuracy@1", 0))
                    mrr10 = float(d.get("eval_geo-dev_cos_sim_mrr@10", 0))
                    recall10 = float(d.get("eval_geo-dev_cos_sim_recall@10", 0))
                    map100 = float(d.get("eval_geo-dev_cos_sim_map@100", 0))
                    eval_data.append({
                        "epoch": epoch,
                        "ndcg10": ndcg10,
                        "acc1": acc1,
                        "mrr10": mrr10,
                        "recall10": recall10,
                        "map100": map100,
                    })
                except Exception:
                    pass

    return loss_data, eval_data


def deduplicate_eval(eval_data):
    """Оставить по одному значению на каждую эпоху (первое вхождение)."""
    seen = set()
    result = []
    for d in eval_data:
        key = round(d["epoch"], 3)
        if key not in seen:
            seen.add(key)
            result.append(d)
    return result


def plot(loss_data, eval_data, output_path: Path):
    eval_data = deduplicate_eval(eval_data)

    loss_epochs   = [d["epoch"]     for d in loss_data]
    losses        = [d["loss"]      for d in loss_data]
    lrs           = [d["lr"]        for d in loss_data]
    grad_norms    = [d["grad_norm"] for d in loss_data]

    eval_epochs   = [d["epoch"]   for d in eval_data]
    ndcg10s       = [d["ndcg10"]  for d in eval_data]
    acc1s         = [d["acc1"]    for d in eval_data]
    mrr10s        = [d["mrr10"]   for d in eval_data]
    recall10s     = [d["recall10"]for d in eval_data]

    best_idx = ndcg10s.index(max(ndcg10s))
    best_epoch = eval_epochs[best_idx]
    best_ndcg  = ndcg10s[best_idx]

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("BGE-M3 Fine-tuning — Training Dynamics", fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    # ── 1. Train Loss ──────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(loss_epochs, losses, color="#2563eb", linewidth=1.2, alpha=0.85, label="Train loss")
    ax1.axvline(best_epoch, color="red", linestyle="--", linewidth=1, alpha=0.7, label=f"Best NDCG@10 (epoch {best_epoch:.2f})")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Train Loss")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    # ── 2. Eval Metrics ────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(eval_epochs, ndcg10s,   marker="o", ms=4, label=f"NDCG@10  (best {best_ndcg:.4f})", color="#16a34a")
    ax2.plot(eval_epochs, acc1s,     marker="s", ms=4, label=f"Acc@1    (best {max(acc1s):.4f})",  color="#dc2626")
    ax2.plot(eval_epochs, mrr10s,    marker="^", ms=4, label=f"MRR@10   (best {max(mrr10s):.4f})", color="#9333ea")
    ax2.plot(eval_epochs, recall10s, marker="D", ms=3, label=f"Recall@10 (best {max(recall10s):.4f})", color="#ea580c", alpha=0.7)
    ax2.axvline(best_epoch, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Score")
    ax2.set_title("Dev Eval Metrics")
    ax2.legend(fontsize=7.5); ax2.grid(alpha=0.3)
    ax2.set_ylim(0.75, 1.0)

    # ── 3. Learning Rate ───────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(loss_epochs, lrs, color="#0891b2", linewidth=1.2)
    ax3.axvline(best_epoch, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax3.set_xlabel("Epoch"); ax3.set_ylabel("Learning Rate")
    ax3.set_title("Learning Rate Schedule")
    ax3.grid(alpha=0.3)
    # Аннотация: LR → 0
    zero_lr_epochs = [e for e, lr in zip(loss_epochs, lrs) if lr == 0.0]
    if zero_lr_epochs:
        ax3.axvline(zero_lr_epochs[0], color="orange", linestyle=":", linewidth=1.5,
                    label=f"LR→0 (epoch {zero_lr_epochs[0]:.2f})")
        ax3.legend(fontsize=8)

    # ── 4. Grad Norm ───────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(loss_epochs, grad_norms, color="#be185d", linewidth=1, alpha=0.75)
    ax4.axvline(best_epoch, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax4.set_xlabel("Epoch"); ax4.set_ylabel("Grad Norm")
    ax4.set_title("Gradient Norm")
    ax4.grid(alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ График сохранён: {output_path}")

    # Печать сводки
    print("\n── Сводка ──────────────────────────────────────────")
    print(f"Всего eval чекпоинтов: {len(eval_data)}")
    print(f"Loss: {losses[0]:.3f} → {losses[-1]:.3f} (−{losses[0]-losses[-1]:.3f})")
    print(f"Лучший NDCG@10: {best_ndcg:.4f} @ epoch {best_epoch:.3f}")
    print(f"Финальный NDCG@10: {ndcg10s[-1]:.4f}")
    if zero_lr_epochs:
        print(f"LR обнулился на epoch {zero_lr_epochs[0]:.2f} → плато метрик")
    print("────────────────────────────────────────────────────")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, default=Path("experiment.txt"))
    parser.add_argument("--output", type=Path, default=Path("output/training_plots.png"))
    args = parser.parse_args()

    loss_data, eval_data = parse_log(args.log)
    print(f"Найдено loss-записей: {len(loss_data)}, eval-чекпоинтов: {len(eval_data)}")

    if not loss_data:
        print("[ERROR] Не найдено loss-данных. Проверьте формат файла.")
        return

    plot(loss_data, eval_data, args.output)


if __name__ == "__main__":
    main()
