import os

import hydra
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import DictConfig
from torch import cuda
import torch.nn as nn
from torchvision.models import EfficientNet

from ai_detector_model.core.config import CONFIG_DIR
from ai_detector_model.data.factory import create_test_loader, create_train_loader
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

USE_AMP = os.getenv("USE_AMP")
match USE_AMP:
    case "true":
        USE_AMP = True
        pass
    case _:
        USE_AMP = False

logger.info(f"USE_AMP = {USE_AMP}")


@hydra.main(config_path=str(CONFIG_DIR), config_name="config", version_base="1.3")
def main(
    cfg: DictConfig,
):
    output_dir = HydraConfig.get().runtime.output_dir
    logger.info("Creating data loaders")

    train_loader = create_train_loader(cfg)
    test_loader = create_test_loader(cfg)

    logger.info("Instantiating objects")

    model = hydra.utils.instantiate(cfg.model.estimator)

    # if you want to use other architecture - create section for swapping head
    # and pulling backbone and head params separately. use appropriate optimizer.
    if cfg.model.model_type == "efficientnet":
        model: EfficientNet
        model.classifier[1] = nn.Linear(
            in_features=model.classifier[1].in_features,
            out_features=cfg.data.out_features,
        )
        backbone_params = [
            param for name, param in model.named_parameters() if "classifier" not in name
        ]
        head_params = model.classifier.parameters()

        optim_partial = hydra.utils.instantiate(cfg.train.optimizer.adamw)
        base_lr = cfg.train.optimizer.adamw.lr

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
        output_dir=output_dir,
        use_amp=USE_AMP,
        metric_name=cfg.train.metric.name,
        metric_mode=cfg.train.metric.mode,
    )
    logger.info("STARTING TRAINING")
    trainer.train(max_epochs=cfg.train.epochs)


if __name__ == "__main__":
    main()
