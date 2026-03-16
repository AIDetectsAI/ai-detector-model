import os

import hydra
from loguru import logger
from omegaconf import DictConfig
from torch import cuda
from tqdm import tqdm

from ai_detector_model.config import CONFIG_DIR

# DEVICE SETUP
DEVICE = os.getenv("TRAIN_DEVICE")
_device_available = True
match DEVICE:
    case "cuda":
        if not cuda.is_available():
            _device_available = False
    case "cpu":
        pass
    case _:
        _device_available = False

if not _device_available:
    logger.warning(f"Device {DEVICE} is not available! Switching to CPU")
    DEVICE = "cpu"

logger.info(f"Training device: {DEVICE}")


@hydra.main(config_path=str(CONFIG_DIR), config_name="config", version_base="1.3")
def main(
    cfg: DictConfig,
):
    logger.info(f"epochs: {cfg.train.epochs}")
    logger.info("Training some model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Modeling training complete.")


if __name__ == "__main__":
    main()
