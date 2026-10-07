"""Treinamento da CNN, com suporte a retomar de onde parou.

Uso:
    python -m module_classificador_imagens.modeling.train
    python -m module_classificador_imagens.modeling.train --epochs 10
    python -m module_classificador_imagens.modeling.train --resume --epochs 80
"""
import json
from pathlib import Path

from loguru import logger
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import typer

from module_classificador_imagens.config import (
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    FIGURES_DIR,
    HISTORY_PATH,
    MODEL_PATH,
    VALIDATION_SPLIT,
)
from module_classificador_imagens.dataset import load_cifar10
from module_classificador_imagens.modeling.model import build_model
from module_classificador_imagens.plots import plot_history

app = typer.Typer()


class HistorySaver(keras.callbacks.Callback):
    """Acumula as métricas de cada época e grava o JSON a cada época.

    Assim o histórico sobrevive a interrupções (Ctrl+C, queda de energia etc.).
    """

    def __init__(self, history: dict, path: Path):
        super().__init__()
        self.history = history
        self.path = path

    def on_epoch_end(self, epoch, logs=None):
        for key, value in (logs or {}).items():
            self.history.setdefault(key, []).append(float(value))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.history, indent=2))


@app.command()
def main(
    epochs: int = EPOCHS,
    validation_split: float = VALIDATION_SPLIT,
    patience: int = EARLY_STOPPING_PATIENCE,
    resume: bool = False,
    model_path: Path = MODEL_PATH,
    history_path: Path = HISTORY_PATH,
):
    """Treina a CNN.

    Com --resume, carrega o modelo e o histórico salvos e continua a partir da
    última época. Nesse caso, --epochs é o TOTAL desejado (ex.: já fez 20 e
    quer chegar a 50 -> --epochs 50).
    """
    (X_train, y_train), _ = load_cifar10()

    history: dict = {}
    initial_epoch = 0
    best_val_loss = None

    if resume:
        if not model_path.exists():
            logger.error(f"Nada para retomar: {model_path} não existe.")
            raise typer.Exit(code=1)

        model = keras.models.load_model(model_path)  # pesos + optimizer + compile

        if history_path.exists():
            history = json.loads(history_path.read_text())
            initial_epoch = len(history.get("loss", []))
            if history.get("val_loss"):
                best_val_loss = min(history["val_loss"])

        if initial_epoch >= epochs:
            logger.warning(
                f"O histórico já tem {initial_epoch} épocas. "
                f"Use --epochs maior que {initial_epoch} para continuar."
            )
            raise typer.Exit()

        logger.info(f"Retomando da época {initial_epoch + 1} até {epochs}.")
    else:
        model = build_model()
        model.summary()
        logger.info("Iniciando treinamento do zero...")

    model_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,  # para se val_loss não melhorar por N épocas seguidas
            restore_best_weights=True,
        ),
        # Salva o melhor modelo durante o treino. O initial_value_threshold evita
        # que, ao retomar, a 1ª época sobrescreva um modelo melhor que já existe.
        ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
            initial_value_threshold=best_val_loss,
        ),
        HistorySaver(history, history_path),
    ]

    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        initial_epoch=initial_epoch,
        validation_split=validation_split,
        callbacks=callbacks,
    )

    model.save(model_path)
    plot_history(history, save_path=FIGURES_DIR / "training_history.png", show=False)
    logger.success(f"Modelo salvo em {model_path} | histórico em {history_path}")


if __name__ == "__main__":
    app()