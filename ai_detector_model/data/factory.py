import albumentations as A
from omegaconf import DictConfig


def get_transforms(cfg: DictConfig, stage="train"):
    image_size = cfg.model.input_size
    tr_cfg = cfg.data.transforms
    if stage == "train":
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(tr_cfg.horizontal_flip_p),
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
