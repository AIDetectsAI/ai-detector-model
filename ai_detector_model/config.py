from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

# Load environment variables from .env file if it exists
load_dotenv()

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
CONFIG_DIR = PROJ_ROOT / "configs"

MODELS_DIR = PROJ_ROOT / "models"
ONNX_MODELS_DIR = MODELS_DIR / "onnx"
PYTORCH_MODELS_DIR = MODELS_DIR / "pytorch"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


# NOTE: Loguru file logging

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
