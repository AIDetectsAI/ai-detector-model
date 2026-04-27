import albumentations as A
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_image_dataloader(
    image_dir: str, image_size: int = 64, batch_size: int = 32, shuffle: bool = False
):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.ImageFolder(root=image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def preprocess_image(img: Image.Image, image_size: int = 64) -> np.ndarray:
    transform = A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.ToTensorV2(),
        ]
    )

    tensor = transform(image=np.array(img))["image"].unsqueeze(0)
    return tensor.numpy()
