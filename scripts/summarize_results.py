"""Aggregate SimCLR experiment metrics and generate publication-ready figures."""

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SEEDS = (41, 42, 43)
METHODS = ("Scratch", "Linear Probe", "Fine-tune")
COLORS = {"Scratch": "#8C8C8C", "Linear Probe": "#4C78A8", "Fine-tune": "#F58518"}
EXPERIMENTS = {
    ("1%", "Scratch"): "scratch_1percent_100epoch_seed{}",
    ("1%", "Linear Probe"): "linear_probe_100ep_1percent_100epoch_seed{}",
    ("1%", "Fine-tune"): "fine_tune_100ep_1percent_100epoch_seed{}",
    ("10%", "Scratch"): "scratch_10percent_100epoch_seed{}",
    ("10%", "Linear Probe"): "linear_probe_100ep_10percent_seed{}",
    ("10%", "Fine-tune"): "fine_tune_100ep_10percent_seed{}",
}
CLASS_NAMES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def mean_std(values):
    return statistics.mean(values), statistics.stdev(values)


def load_runs(project_root):
    root = project_root / "outputs" / "classification"
    runs = {}
    for key, pattern in EXPERIMENTS.items():
        entries = []
        for seed in SEEDS:
            directory = root / pattern.format(seed)
            test = json.loads((directory / "test_metrics.json").read_text(encoding="utf-8"))
            validation = json.loads((directory / "best_metrics.json").read_text(encoding="utf-8"))
            with (directory / "train_log.csv").open(newline="", encoding="utf-8") as file:
                log = list(csv.DictReader(file))
            entries.append({"seed": seed, "test": test, "validation": validation, "log": log})
        runs[key] = entries
    return runs


def write_tables(runs, output_dir):
    rows = []
    for labels in ("1%", "10%"):
        for method in METHODS:
            entries = runs[(labels, method)]
            accuracies = [100 * item["test"]["test_accuracy"] for item in entries]
            f1_scores = [100 * item["test"]["test_macro_f1"] for item in entries]
            acc_mean, acc_std = mean_std(accuracies)
            f1_mean, f1_std = mean_std(f1_scores)
            rows.append({
                "labels": labels,
                "method": method,
                "seed_41_accuracy": accuracies[0],
                "seed_42_accuracy": accuracies[1],
                "seed_43_accuracy": accuracies[2],
                "accuracy_mean": acc_mean,
                "accuracy_std": acc_std,
                "macro_f1_mean": f1_mean,
                "macro_f1_std": f1_std,
            })

    with (output_dir / "test_results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# CIFAR-10 test results",
        "",
        "Mean and sample standard deviation over downstream training seeds 41, 42, and 43. All runs use the same seed-42 SimCLR encoder and the same seed-42 data split.",
        "",
        "| Labels | Method | Seed 41 | Seed 42 | Seed 43 | Accuracy (mean ± std) | Macro-F1 (mean ± std) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['labels']} | {row['method']} | {row['seed_41_accuracy']:.2f}% | "
            f"{row['seed_42_accuracy']:.2f}% | {row['seed_43_accuracy']:.2f}% | "
            f"**{row['accuracy_mean']:.2f} ± {row['accuracy_std']:.2f}%** | "
            f"**{row['macro_f1_mean']:.2f} ± {row['macro_f1_std']:.2f}%** |"
        )
    (output_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def style_axis(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", alpha=0.2)


def plot_test_accuracy(runs, figures):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for axis, labels in zip(axes, ("1%", "10%")):
        means, errors = [], []
        for method in METHODS:
            values = [100 * item["test"]["test_accuracy"] for item in runs[(labels, method)]]
            mean, std = mean_std(values)
            means.append(mean)
            errors.append(std)
        bars = axis.bar(METHODS, means, yerr=errors, capsize=4, color=[COLORS[m] for m in METHODS])
        axis.set_title(f"{labels} labels")
        axis.set_ylim(35, 84)
        axis.set_ylabel("Test accuracy (%)")
        axis.tick_params(axis="x", rotation=15)
        style_axis(axis)
        for bar, value in zip(bars, means):
            axis.text(bar.get_x() + bar.get_width() / 2, value + 1.2, f"{value:.2f}", ha="center", fontsize=9)
    fig.suptitle("CIFAR-10 low-label evaluation (mean ± std, 3 downstream seeds)")
    fig.tight_layout()
    fig.savefig(figures / "test_accuracy.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pretraining(project_root, figures):
    path = project_root / "outputs" / "pretrain" / "resnet18_100epoch_seed42" / "train_log.csv"
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    epoch = np.array([int(row["epoch"]) for row in rows])
    loss = np.array([float(row["loss"]) for row in rows])
    positive = np.array([float(row["positive_similarity"]) for row in rows])
    negative = np.array([float(row["negative_similarity"]) for row in rows])
    embedding_std = np.array([float(row["embedding_std"]) for row in rows])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    axes[0].plot(epoch, loss, color="#E45756", linewidth=2)
    axes[0].set(title="NT-Xent loss", xlabel="Epoch", ylabel="Loss")
    axes[1].plot(epoch, positive, label="Positive similarity", linewidth=2)
    axes[1].plot(epoch, negative, label="Negative similarity", linewidth=2)
    axes[1].plot(epoch, embedding_std, label="Embedding std", linewidth=2)
    axes[1].set(title="Representation diagnostics", xlabel="Epoch", ylabel="Value")
    axes[1].legend(frameon=False, fontsize=9)
    for axis in axes:
        style_axis(axis)
    fig.suptitle("SimCLR pretraining on 45,000 unlabeled CIFAR-10 images")
    fig.tight_layout()
    fig.savefig(figures / "pretraining_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_downstream_curves(runs, figures):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for axis, labels in zip(axes, ("1%", "10%")):
        for method in METHODS:
            logs = runs[(labels, method)]
            values = [np.array([100 * float(row["val_accuracy"]) for row in item["log"]]) for item in logs]
            length = min(map(len, values))
            matrix = np.stack([value[:length] for value in values])
            epoch = np.arange(1, length + 1)
            mean = matrix.mean(axis=0)
            std = matrix.std(axis=0, ddof=1)
            axis.plot(epoch, mean, color=COLORS[method], label=method, linewidth=2)
            axis.fill_between(epoch, mean - std, mean + std, color=COLORS[method], alpha=0.13)
        axis.set(title=f"{labels} labels", xlabel="Epoch", ylabel="Validation accuracy (%)")
        style_axis(axis)
    axes[0].legend(frameon=False)
    fig.suptitle("Downstream validation curves (mean ± std, 3 seeds)")
    fig.tight_layout()
    fig.savefig(figures / "downstream_training_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(runs, labels, method, filename, figures):
    matrices = [np.array(item["test"]["confusion_matrix"], dtype=float) for item in runs[(labels, method)]]
    matrix = np.sum(matrices, axis=0)
    normalized = 100 * matrix / matrix.sum(axis=1, keepdims=True)
    fig, axis = plt.subplots(figsize=(7.4, 6.3))
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=100)
    axis.set_xticks(range(10), CLASS_NAMES, rotation=45, ha="right")
    axis.set_yticks(range(10), CLASS_NAMES)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title(f"{labels} labels · {method}\nrow-normalized test confusion matrix, aggregated over 3 seeds")
    for row in range(10):
        for column in range(10):
            value = normalized[row, column]
            if value >= 5 or row == column:
                axis.text(column, row, f"{value:.0f}", ha="center", va="center", fontsize=7,
                          color="white" if value > 55 else "#202020")
    fig.colorbar(image, ax=axis, label="Percentage of true class (%)", fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(figures / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = (args.output_dir or project_root / "results").resolve()
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    runs = load_runs(project_root)
    write_tables(runs, output_dir)
    plot_test_accuracy(runs, figures)
    plot_pretraining(project_root, figures)
    plot_downstream_curves(runs, figures)
    plot_confusion(runs, "1%", "Linear Probe", "confusion_matrix_1percent_linear_probe.png", figures)
    plot_confusion(runs, "10%", "Fine-tune", "confusion_matrix_10percent_fine_tune.png", figures)
    print(f"Wrote tables and figures to {output_dir}")


if __name__ == "__main__":
    main()
