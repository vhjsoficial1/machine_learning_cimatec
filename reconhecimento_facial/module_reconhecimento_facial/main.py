"""CLI do projeto.

Exemplos:
    python -m module_reconhecimento_facial.main testar                 # menu: escolhe a imagem
    python -m module_reconhecimento_facial.main testar caminho/foto.jpg
    python -m module_reconhecimento_facial.main webcam
"""

from pathlib import Path
from typing import Optional

import typer

from module_reconhecimento_facial.config import AUTORIZADOS_DIR, RESULTADOS_DIR, TESTE_DIR
from module_reconhecimento_facial.reconhecimento import ReconhecedorFacial

app = typer.Typer(add_completion=False, help="Reconhecimento facial com OpenCV + DeepFace.")


@app.command()
def testar(
    imagem: Optional[Path] = typer.Argument(
        None, help="Imagem específica. Se omitida, pergunta qual imagem da pasta de teste usar."
    ),
    pasta_teste: Path = typer.Option(TESTE_DIR, help="Pasta com as fotos de teste."),
    pasta_autorizados: Path = typer.Option(AUTORIZADOS_DIR, help="Pasta com fotos autorizadas."),
    saida: Path = typer.Option(RESULTADOS_DIR, help="Pasta para as imagens anotadas."),
    modelo: str = typer.Option("Facenet512", help="Modelo do DeepFace."),
    detector: str = typer.Option("retinaface", help="Detector de rostos do DeepFace."),
    limiar: Optional[float] = typer.Option(None, help="Distância cosseno máxima p/ liberar."),
):
    """Escolhe uma imagem de teste e mostra o resultado com os bounding boxes numa janela."""
    rec = ReconhecedorFacial(pasta_autorizados, modelo, detector, limiar)
    if imagem:
        rec.processar_imagem(imagem, saida)
    else:
        rec.menu_interativo(pasta_teste, saida)


@app.command()
def webcam(
    pasta_autorizados: Path = typer.Option(AUTORIZADOS_DIR),
    modelo: str = typer.Option("Facenet512"),
    detector: str = typer.Option("retinaface", help="Use 'ssd' ou 'opencv' para caixas mais ágeis."),
    limiar: Optional[float] = typer.Option(None),
    camera: int = typer.Option(0, help="Índice da câmera."),
    largura_inferencia: int = typer.Option(
        640, help="Largura do frame enviado ao DeepFace (menor = mais rápido)."
    ),
    espelhar: bool = typer.Option(True, help="Espelha a imagem (efeito espelho)."),
):
    """Reconhecimento em tempo real pela webcam (vídeo fluido, análise em segundo plano)."""
    rec = ReconhecedorFacial(pasta_autorizados, modelo, detector, limiar)
    rec.webcam(camera, largura_inferencia, espelhar=espelhar)


if __name__ == "__main__":
    app()