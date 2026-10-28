from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Carrega variáveis de ambiente do arquivo .env, se existir
load_dotenv()

# Caminhos do projeto
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# --- Reconhecimento facial ---
# Fotos das pessoas autorizadas (uma subpasta por pessoa: autorizados/<nome>/foto1.jpg)
AUTORIZADOS_DIR = RAW_DATA_DIR / "autorizados"
# Fotos para testar o sistema
TESTE_DIR = RAW_DATA_DIR / "teste"
# Imagens de saída com os bounding boxes desenhados
RESULTADOS_DIR = PROCESSED_DATA_DIR / "resultados"

# Se tqdm estiver instalado, configura o loguru para não quebrar as barras de progresso
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except (ModuleNotFoundError, ValueError):
    pass