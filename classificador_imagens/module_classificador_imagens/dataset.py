"""Carregamento da base CIFAR-10."""
from pathlib import Path

from loguru import logger
import numpy as np
from tensorflow import keras
import typer

from module_classificador_imagens.config import PROCESSED_DATA_DIR
from module_classificador_imagens.features import flatten_labels, normalize_images

app = typer.Typer()


def load_cifar10():
    """Retorna ((X_train, y_train), (X_test, y_test)) já pré-processados."""
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

    X_train, X_test = normalize_images(X_train), normalize_images(X_test)
    y_train, y_test = flatten_labels(y_train), flatten_labels(y_test)

    logger.info(f"X_train: {X_train.shape} | y_train: {y_train.shape}")
    logger.info(f"X_test:  {X_test.shape} | y_test:  {y_test.shape}")
    return (X_train, y_train), (X_test, y_test)


@app.command()
def main(output_path: Path = PROCESSED_DATA_DIR / "cifar10.npz"):
    """Baixa o CIFAR-10, pré-processa e salva em data/processed."""
    (X_train, y_train), (X_test, y_test) = load_cifar10()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    logger.success(f"Dataset salvo em {output_path}")


if __name__ == "__main__":
    app()