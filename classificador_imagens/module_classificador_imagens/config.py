from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Carrega variáveis de ambiente do arquivo .env, se existir
load_dotenv()

# -------------------------
# Caminhos
# -------------------------
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

MODEL_PATH = MODELS_DIR / "cnn_cifar10.keras"
HISTORY_PATH = MODELS_DIR / "history.json"

# -------------------------
# Dados
# -------------------------
IMAGE_SHAPE = (32, 32, 3)
NUM_CLASSES = 10
CLASS_NAMES = [
    "avião", "automóvel", "pássaro", "gato", "cervo",
    "cachorro", "sapo", "cavalo", "navio", "caminhão",
]

# -------------------------
# Treinamento
# -------------------------
EPOCHS = 50
VALIDATION_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 3

# Se tqdm estiver instalado, faz o loguru não quebrar as barras de progresso
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except (ModuleNotFoundError, ValueError):
    pass