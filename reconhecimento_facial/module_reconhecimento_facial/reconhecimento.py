"""Reconhecimento facial com OpenCV (desenho/IO/janelas) e DeepFace (detecção + embeddings)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import sys
import threading
import time

import cv2
from deepface import DeepFace
from loguru import logger
import numpy as np

from module_reconhecimento_facial.config import AUTORIZADOS_DIR, RESULTADOS_DIR, TESTE_DIR

EXTENSOES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Cores em BGR (padrão do OpenCV)
VERDE = (0, 255, 0)
VERMELHO = (0, 0, 255)
TEXTO_LIBERADO = "ACESSO LIBERADO"
TEXTO_NEGADO = "ACESSO NEGADO"
FONTE = cv2.FONT_HERSHEY_SIMPLEX

# Limiares de distância cosseno recomendados pelo DeepFace para cada modelo
LIMIARES = {
    "VGG-Face": 0.68,
    "Facenet": 0.40,
    "Facenet512": 0.30,
    "ArcFace": 0.68,
    "SFace": 0.593,
}


@dataclass
class Resultado:
    caixa: tuple[int, int, int, int]  # x, y, largura, altura
    pessoa: str | None
    distancia: float
    liberado: bool


# ---------------------------------------------------------------------- utilitários
def ler_imagem(caminho: Path) -> np.ndarray | None:
    """Lê imagem de forma segura para caminhos com acentos (Windows)."""
    dados = np.fromfile(str(caminho), dtype=np.uint8)
    if dados.size == 0:
        return None
    return cv2.imdecode(dados, cv2.IMREAD_COLOR)


def salvar_imagem(caminho: Path, imagem: np.ndarray) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(caminho.suffix or ".jpg", imagem)
    if not ok:
        raise IOError(f"Não foi possível codificar a imagem {caminho}")
    buffer.tofile(str(caminho))


def listar_imagens(pasta: Path) -> list[Path]:
    return sorted(p for p in Path(pasta).rglob("*") if p.suffix.lower() in EXTENSOES)


def _ajustar_para_tela(imagem: np.ndarray, max_w: int = 1100, max_h: int = 750) -> np.ndarray:
    """Reduz a imagem só para exibir na janela (o arquivo salvo mantém a resolução original)."""
    h, w = imagem.shape[:2]
    escala = min(1.0, max_w / w, max_h / h)
    if escala < 1.0:
        return cv2.resize(imagem, (int(w * escala), int(h * escala)), interpolation=cv2.INTER_AREA)
    return imagem


def _janela_aberta(titulo: str) -> bool:
    try:
        return cv2.getWindowProperty(titulo, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


def _mostrar(titulo: str, imagem: np.ndarray) -> None:
    """Abre uma janela com a imagem. Fecha com qualquer tecla ou no X da janela."""
    cv2.imshow(titulo, _ajustar_para_tela(imagem))
    while True:
        if cv2.waitKey(50) != -1 or not _janela_aberta(titulo):
            break
    cv2.destroyWindow(titulo)
    cv2.waitKey(1)


class ReconhecedorFacial:
    def __init__(
        self,
        pasta_autorizados: Path = AUTORIZADOS_DIR,
        modelo: str = "Facenet512",
        detector: str = "retinaface",
        limiar: float | None = None,
    ):
        self.modelo = modelo
        self.detector = detector
        self.limiar = limiar if limiar is not None else LIMIARES.get(modelo, 0.40)
        self._nomes: list[str] = []
        self._embeddings = np.empty((0, 0))
        self.carregar_autorizados(pasta_autorizados)

    # ------------------------------------------------------------------ base
    def carregar_autorizados(self, pasta: Path) -> None:
        """Gera o embedding de cada foto de pessoa autorizada."""
        pasta = Path(pasta)
        nomes, embeddings = [], []

        for caminho in listar_imagens(pasta):
            imagem = ler_imagem(caminho)
            if imagem is None:
                logger.warning(f"Imagem ilegível: {caminho}")
                continue

            faces = self._extrair_faces(imagem)
            if not faces:
                logger.warning(f"Nenhum rosto encontrado em: {caminho.name}")
                continue
            if len(faces) > 1:
                logger.warning(f"{caminho.name}: mais de um rosto, usando o maior")

            face = max(faces, key=lambda f: f["facial_area"]["w"] * f["facial_area"]["h"])
            emb = np.asarray(face["embedding"], dtype=np.float64)
            embeddings.append(emb / np.linalg.norm(emb))

            # Subpasta = nome da pessoa; foto solta na raiz = nome do arquivo
            nomes.append(caminho.parent.name if caminho.parent != pasta else caminho.stem)

        self._nomes = nomes
        self._embeddings = np.vstack(embeddings) if embeddings else np.empty((0, 0))
        logger.info(
            f"{len(nomes)} foto(s) autorizada(s) carregada(s) "
            f"({len(set(nomes))} pessoa(s)) | modelo={self.modelo} limiar={self.limiar}"
        )

    def _extrair_faces(self, imagem: np.ndarray) -> list[dict]:
        try:
            return DeepFace.represent(
                img_path=imagem,
                model_name=self.modelo,
                detector_backend=self.detector,
                enforce_detection=True,
                align=True,
            )
        except ValueError:  # DeepFace levanta ValueError quando não há rosto
            return []

    # ----------------------------------------------------------- identificação
    def identificar(self, imagem: np.ndarray) -> list[Resultado]:
        """Detecta todos os rostos da imagem e decide acesso para cada um."""
        resultados = []
        for face in self._extrair_faces(imagem):
            area = face["facial_area"]
            caixa = (area["x"], area["y"], area["w"], area["h"])

            pessoa, distancia, liberado = None, 1.0, False
            if len(self._nomes) > 0:
                emb = np.asarray(face["embedding"], dtype=np.float64)
                emb /= np.linalg.norm(emb)
                distancias = 1.0 - self._embeddings @ emb  # distância cosseno
                melhor = int(np.argmin(distancias))
                distancia = float(distancias[melhor])
                if distancia <= self.limiar:
                    pessoa, liberado = self._nomes[melhor], True

            resultados.append(Resultado(caixa, pessoa, distancia, liberado))
        return resultados

    # ------------------------------------------------------------------ desenho
    @staticmethod
    def desenhar(imagem: np.ndarray, resultados: list[Resultado]) -> np.ndarray:
        """Bounding box VERDE + 'ACESSO LIBERADO' ou VERMELHO + 'ACESSO NEGADO'."""
        saida = imagem.copy()
        altura, largura = saida.shape[:2]
        escala = max(0.5, max(altura, largura) / 1200)
        espessura = max(2, int(round(escala * 2)))
        pad = 6

        for r in resultados:
            x, y, w, h = r.caixa
            cor = VERDE if r.liberado else VERMELHO
            cor_texto = (0, 0, 0) if r.liberado else (255, 255, 255)
            texto = TEXTO_LIBERADO if r.liberado else TEXTO_NEGADO

            cv2.rectangle(saida, (x, y), (x + w, y + h), cor, espessura)

            (tw, th), base = cv2.getTextSize(texto, FONTE, escala, espessura)
            topo = y - th - base - 2 * pad
            if topo < 0:  # sem espaço acima: desenha o rótulo dentro da caixa
                topo = y
            cv2.rectangle(
                saida, (x, topo), (x + tw + 2 * pad, topo + th + base + 2 * pad), cor, cv2.FILLED
            )
            cv2.putText(
                saida, texto, (x + pad, topo + th + pad), FONTE, escala, cor_texto,
                espessura, cv2.LINE_AA,
            )

            if r.liberado and r.pessoa:
                y_nome = min(y + h + th + pad, altura - 5)
                cv2.putText(
                    saida, r.pessoa, (x, y_nome), FONTE, escala, VERDE, espessura, cv2.LINE_AA
                )
        return saida

    # ------------------------------------------------------- imagens (interativo)
    def processar_imagem(
        self, caminho: Path, pasta_saida: Path = RESULTADOS_DIR, mostrar: bool = True
    ) -> np.ndarray | None:
        """Analisa uma imagem, salva o resultado e (opcionalmente) abre a janela com os boxes."""
        caminho = Path(caminho)
        imagem = ler_imagem(caminho) if caminho.is_file() else None
        if imagem is None:
            logger.error(f"Não foi possível abrir a imagem: {caminho}")
            return None

        resultados = self.identificar(imagem)
        if not resultados:
            logger.warning(f"{caminho.name}: nenhum rosto detectado")
        for r in resultados:
            status = TEXTO_LIBERADO if r.liberado else TEXTO_NEGADO
            logger.info(f"{caminho.name}: {status} | pessoa={r.pessoa} dist={r.distancia:.3f}")

        anotada = self.desenhar(imagem, resultados)
        destino = Path(pasta_saida) / f"{caminho.stem}_resultado{caminho.suffix}"
        salvar_imagem(destino, anotada)
        logger.info(f"Resultado salvo em {destino}")

        if mostrar:
            _mostrar(f"Resultado - {caminho.name} (qualquer tecla fecha)", anotada)
        return anotada

    def menu_interativo(self, pasta_teste: Path = TESTE_DIR, pasta_saida: Path = RESULTADOS_DIR) -> None:
        """Pergunta qual imagem usar e mostra o resultado numa janela. Repete até sair."""
        pasta_teste = Path(pasta_teste)
        imagens = listar_imagens(pasta_teste)
        if not imagens:
            logger.warning(f"Nenhuma imagem em {pasta_teste}")
            return

        while True:
            print("\nImagens disponíveis em", pasta_teste)
            for i, p in enumerate(imagens, 1):
                print(f"  [{i}] {p.relative_to(pasta_teste)}")
            print("  [t] todas    [q] sair")

            try:
                escolha = input("Qual imagem usar? (número, caminho ou t/q): ").strip().strip("\"'")
            except (EOFError, KeyboardInterrupt):
                print()
                return

            if escolha.lower() in {"q", "s", "sair", ""}:
                return
            if escolha.lower() in {"t", "todas"}:
                for p in imagens:
                    self.processar_imagem(p, pasta_saida)
                continue
            if escolha.isdigit() and 1 <= int(escolha) <= len(imagens):
                self.processar_imagem(imagens[int(escolha) - 1], pasta_saida)
                continue
            if Path(escolha).is_file():  # caminho completo (ex.: arrastar o arquivo p/ o terminal)
                self.processar_imagem(Path(escolha), pasta_saida)
                continue
            print("Opção inválida.")

    # ------------------------------------------------------------------- webcam
    def webcam(
        self,
        camera: int = 0,
        largura_inferencia: int = 640,
        validade: float = 1.5,
        espelhar: bool = True,
    ) -> None:
        """Reconhecimento em tempo real.

        O vídeo é exibido na velocidade da câmera; o DeepFace roda em uma thread separada,
        sempre sobre o frame mais recente e reduzido, então a imagem não trava.
        Pressione 'q' ou ESC (ou feche a janela) para sair.
        """
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(camera, backend)
        if not cap.isOpened():
            logger.error(f"Não foi possível abrir a câmera {camera}")
            return
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # evita atraso por acúmulo de frames

        lock = threading.Lock()
        parar = threading.Event()
        estado: dict = {"frame": None, "resultados": [], "instante": 0.0, "latencia": 0.0}

        def trabalhador() -> None:
            while not parar.is_set():
                with lock:
                    frame, estado["frame"] = estado["frame"], None
                if frame is None:
                    time.sleep(0.005)
                    continue

                inicio = time.time()
                altura, largura = frame.shape[:2]
                escala = min(1.0, largura_inferencia / largura)
                pequeno = (
                    cv2.resize(frame, (int(largura * escala), int(altura * escala)),
                               interpolation=cv2.INTER_AREA)
                    if escala < 1.0 else frame
                )
                try:
                    achados = self.identificar(pequeno)
                except Exception as erro:  # a webcam não pode cair por falha numa análise
                    logger.error(f"Falha na análise: {erro}")
                    achados = []

                f = 1.0 / escala
                achados = [
                    replace(r, caixa=tuple(int(v * f) for v in r.caixa)) for r in achados
                ]
                with lock:
                    estado["resultados"] = achados
                    estado["instante"] = time.time()
                    estado["latencia"] = estado["instante"] - inicio

        fio = threading.Thread(target=trabalhador, daemon=True)
        fio.start()

        titulo = "Reconhecimento facial (q para sair)"
        ultimo, fps = time.time(), 0.0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if espelhar:
                    frame = cv2.flip(frame, 1)

                with lock:
                    estado["frame"] = frame  # o trabalhador pega sempre o mais recente
                    resultados = (
                        estado["resultados"] if time.time() - estado["instante"] <= validade else []
                    )
                    latencia = estado["latencia"]

                exibir = self.desenhar(frame, resultados)

                agora = time.time()
                fps = 0.9 * fps + 0.1 / max(agora - ultimo, 1e-6)
                ultimo = agora
                cv2.putText(
                    exibir, f"FPS {fps:4.0f} | analise {latencia * 1000:4.0f} ms", (10, 25),
                    FONTE, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
                )

                cv2.imshow(titulo, exibir)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27) or not _janela_aberta(titulo):
                    break
        finally:
            parar.set()
            fio.join(timeout=5)
            cap.release()
            cv2.destroyAllWindows()