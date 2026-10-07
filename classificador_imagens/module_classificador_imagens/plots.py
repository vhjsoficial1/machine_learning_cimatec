"""Visualizações. Cada função pode salvar a figura em disco e/ou exibi-la."""
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from module_classificador_imagens.config import CLASS_NAMES


def _finish(save_path: Optional[Path], show: bool) -> None:
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_samples(X, y, n: int = 9, save_path: Optional[Path] = None, show: bool = True):
    """Grade 3x3 com exemplos da base."""
    plt.figure(figsize=(10, 10))
    for i in range(n):
        plt.subplot(3, 3, i + 1)
        plt.imshow(X[i])
        plt.title(CLASS_NAMES[y[i]])
        plt.axis("off")
    plt.suptitle("Exemplos da base CIFAR-10")
    _finish(save_path, show)


def plot_history(history: dict, save_path: Optional[Path] = None, show: bool = True):
    """Curvas de acurácia e loss (treino x validação)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, title in zip(
        axes, ["accuracy", "loss"], ["Acurácia por época", "Loss por época"]
    ):
        ax.plot(history[metric], label="Treino")
        ax.plot(history[f"val_{metric}"], label="Validação")
        ax.set_title(title)
        ax.set_xlabel("Épocas")
        ax.set_ylabel(metric.capitalize())
        ax.legend()

    _finish(save_path, show)


def plot_predictions(
    X, y_true, y_pred, n: int = 9, save_path: Optional[Path] = None, show: bool = True
):
    """Grade 3x3 comparando rótulo real x predição."""
    plt.figure(figsize=(10, 10))
    for i in range(n):
        plt.subplot(3, 3, i + 1)
        plt.imshow(X[i])
        plt.title(f"Real: {CLASS_NAMES[y_true[i]]}\nPredição: {CLASS_NAMES[y_pred[i]]}")
        plt.axis("off")
    plt.suptitle("Previsões em imagens do conjunto de teste")
    _finish(save_path, show)