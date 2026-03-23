import os.path
from typing import Any

import albumentations as A
from omegaconf import DictConfig
from torch.utils.data import DataLoader

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


def create_train_loader(cfg: DictConfig) -> DataLoader[Any]:
    train_transform = create_transforms(cfg, "train")
    train_path = os.path.join(cfg.data.dataset.path, "train")
    train_dataset = ClassificationDataset(train_path, train_transform)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.data.dataset.batch_size,
        num_workers=cfg.data.dataset.num_workers,
        pin_memory=cfg.data.dataset.pin_memory,
        shuffle=cfg.data.dataset.shuffle,
        prefetch_factor=cfg.data.dataset.prefetch_factor,
        drop_last=True
    )

    return train_loader


def create_test_loader(cfg: DictConfig) -> DataLoader[Any]:
    test_transform = create_transforms(cfg, "test")
    test_path = os.path.join(cfg.data.dataset.path, "test")
    test_dataset = ClassificationDataset(test_path, test_transform)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=cfg.data.dataset.batch_size * 2,
        num_workers=cfg.data.dataset.num_workers,
        pin_memory=cfg.data.dataset.pin_memory,
        prefetch_factor=cfg.data.dataset.prefetch_factor,
    )

    return test_loader
