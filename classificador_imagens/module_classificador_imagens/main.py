"""Ponto de entrada do projeto.

Uso:
    python -m module_classificador_imagens.main all          # pipeline completo
    python -m module_classificador_imagens.main train        # só treina
    python -m module_classificador_imagens.main train --resume --epochs 80
    python -m module_classificador_imagens.main history      # replota métricas salvas
    python -m module_classificador_imagens.main evaluate     # só avalia
    python -m module_classificador_imagens.main samples      # grade de previsões
    python -m module_classificador_imagens.main image foto.jpg
    python -m module_classificador_imagens.main --help
"""
import json
from pathlib import Path

from loguru import logger
import typer

from module_classificador_imagens.config import FIGURES_DIR, HISTORY_PATH
from module_classificador_imagens.dataset import load_cifar10
from module_classificador_imagens.modeling import predict, train
from module_classificador_imagens.plots import plot_history, plot_samples

app = typer.Typer(help="Classificador de imagens CIFAR-10 (CNN).")

# Reaproveita os comandos já definidos em train.py e predict.py
app.command("train")(train.main)
app.command("evaluate")(predict.evaluate)
app.command("samples")(predict.samples)
app.command("image")(predict.image)


@app.command("history")
def show_history(history_path: Path = HISTORY_PATH):
    """Replota acurácia/loss a partir do histórico salvo, sem retreinar."""
    if not history_path.exists():
        logger.error(f"Histórico não encontrado: {history_path}")
        raise typer.Exit(code=1)

    history = json.loads(history_path.read_text())
    logger.info(
        f"{len(history['loss'])} épocas | "
        f"melhor val_accuracy: {max(history['val_accuracy']):.2%} | "
        f"menor val_loss: {min(history['val_loss']):.4f}"
    )
    plot_history(history, save_path=FIGURES_DIR / "training_history.png", show=True)


@app.command("all")
def run_all(epochs: int = train.EPOCHS):
    """Executa tudo: exemplos da base -> treino -> avaliação -> previsões."""
    logger.info("1/4 Gerando exemplos da base...")
    (X_train, y_train), _ = load_cifar10()
    plot_samples(X_train, y_train, save_path=FIGURES_DIR / "cifar10_samples.png", show=False)

    logger.info("2/4 Treinando...")
    train.main(epochs=epochs)

    logger.info("3/4 Avaliando no conjunto de teste...")
    predict.evaluate()

    logger.info("4/4 Gerando previsões de exemplo...")
    predict.samples()

    logger.success("Pipeline concluído.")


if __name__ == "__main__":
    app()