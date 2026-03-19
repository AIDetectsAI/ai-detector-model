import os

import hydra
from loguru import logger
from omegaconf import DictConfig
from torch import cuda
import torch.nn as nn
from torchvision.models import EfficientNet

from ai_detector_model.core.config import CONFIG_DIR
from ai_detector_model.data.factory import create_loaders
from ai_detector_model.modeling.engine import Trainer

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
    logger.info("Creating data loaders")
    train_loader, test_loader = create_loaders(cfg)

    logger.info("Instantiating objects")

    model = hydra.utils.instantiate(cfg.model.estimator)

    if cfg.model.model_type == "efficientnet":
        model: EfficientNet
        model.classifier[1] = nn.Linear(
            in_features=model.classifier[1].in_features,
            out_features=cfg.data.num_classes,
        )
        backbone_params = [
            param for name, param in model.named_parameters() if "classifier" not in name
        ]
        head_params = model.classifier.parameters()

        optim_partial = hydra.utils.instantiate(cfg.train.optimizer.adamw)
        base_lr = cfg.optimizer.adamw.lr

    else:
        raise ValueError("Model type unknown")

    model = model.to(device=DEVICE)

    loss_fn = hydra.utils.instantiate(cfg.train.loss_function)
    optim = optim_partial(
        [
            {"params": backbone_params, "lr": base_lr * 0.1},
            {"params": head_params},
        ]
    )

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optim=optim,
        train_loader=train_loader,
        test_loader=test_loader,
        device=DEVICE,
    )
    trainer.train()

    logger.info("Starting model training")
    logger.success("Modeling training complete.")


if __name__ == "__main__":
    main()
