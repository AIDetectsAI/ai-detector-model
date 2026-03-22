import os.path

import albumentations as A
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from ai_detector_model.core.config import PROCESSED_DATA_DIR
from ai_detector_model.data.dataset import ClassificationDataset


def create_transforms(cfg: DictConfig, stage="train") -> A.Compose:
    image_size = cfg.model.input_size
    tr_cfg = cfg.data.dataset.transforms
    if stage == "train":
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=tr_cfg.horizontal_flip_p),
                A.VerticalFlip(p=tr_cfg.vertical_flip_p),
                A.Rotate(limit=tr_cfg.rotation, p=0.5),
                A.Normalize(mean=tr_cfg.normalize.mean, std=tr_cfg.normalize.std),
                A.ToTensorV2(),
            ]
        )
    elif stage == "test":
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.Normalize(mean=tr_cfg.normalize.mean, std=tr_cfg.normalize.std),
                A.ToTensorV2(),
            ]
        )
    raise ValueError("Wrong stage has been specified")


def create_loaders(cfg: DictConfig) -> tuple[DataLoader, DataLoader]:
    train_transform = create_transforms(cfg, "train")
    test_transform = create_transforms(cfg, "test")

    train_path = os.path.join(PROCESSED_DATA_DIR, "train")
    test_path = os.path.join(PROCESSED_DATA_DIR, "test")

    train_dataset = ClassificationDataset(train_path, train_transform)
    test_dataset = ClassificationDataset(test_path, test_transform)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.data.dataset.batch_size,
        num_workers=cfg.data.dataset.num_workers,
        pin_memory=True,
        shuffle=False,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=cfg.data.dataset.batch_size,
        num_workers=cfg.data.dataset.num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader
