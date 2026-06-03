"""Plota o histograma de classes do arquivo ISIC2018_Task3_Training_GroundTruth.csv."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]


def plot_ground_truth_histogram(csv_path: Path | str = "ISIC2018_Task3_Training_GroundTruth.csv") -> None:
    """Lê o CSV de ground truth e plota a contagem das classes."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV não encontrado: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [c for c in CLASSES if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no CSV: {missing}")

    counts = df[CLASSES].sum(axis=0).astype(int)

    plt.figure(figsize=(10, 5))
    bars = plt.bar(CLASSES, counts, color="tab:blue", edgecolor="black")
    plt.title("Distribuição das Classes no ISIC 2018 Task 3")
    plt.xlabel("Classe")
    plt.ylabel("Quantidade de imagens")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for bar in bars:
        height = bar.get_height()
        plt.annotate(
            f"{int(height)}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_ground_truth_histogram()