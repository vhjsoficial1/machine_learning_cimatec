"""Avaliação e inferência.

Uso:
    python -m module_classificador_imagens.modeling.predict evaluate
    python -m module_classificador_imagens.modeling.predict samples
    python -m module_classificador_imagens.modeling.predict image caminho/foto.jpg
"""
from pathlib import Path

from loguru import logger
import numpy as np
from tensorflow import keras
import typer

from module_classificador_imagens.config import CLASS_NAMES, FIGURES_DIR, MODEL_PATH
from module_classificador_imagens.dataset import load_cifar10
from module_classificador_imagens.features import load_image_file
from module_classificador_imagens.plots import plot_predictions

app = typer.Typer()


@app.command()
def evaluate(model_path: Path = MODEL_PATH):
    """Calcula loss e acurácia no conjunto de teste."""
    _, (X_test, y_test) = load_cifar10()
    model = keras.models.load_model(model_path)
    test_loss, test_acc = model.evaluate(X_test, y_test)
    logger.info(f"Acurácia no teste: {test_acc:.2%}")
    logger.info(f"Loss no teste: {test_loss:.4f}")


@app.command()
def samples(model_path: Path = MODEL_PATH, n: int = 9, show: bool = False):
    """Gera a grade de previsões em imagens do teste."""
    _, (X_test, y_test) = load_cifar10()
    model = keras.models.load_model(model_path)

    pred_labels = np.argmax(model.predict(X_test[:n]), axis=1)
    out = FIGURES_DIR / "test_predictions.png"
    plot_predictions(X_test, y_test, pred_labels, n=n, save_path=out, show=show)
    logger.success(f"Figura salva em {out}")


@app.command()
def image(path: Path, model_path: Path = MODEL_PATH):
    """Classifica uma imagem qualquer do disco."""
    model = keras.models.load_model(model_path)
    probs = model.predict(load_image_file(path))[0]
    idx = int(np.argmax(probs))
    logger.info(f"{path.name}: {CLASS_NAMES[idx]} ({probs[idx]:.1%})")


if __name__ == "__main__":
    app()