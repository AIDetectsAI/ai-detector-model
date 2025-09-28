from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from PIL import Image

def get_image_dataloader(image_dir: str, image_size: int = 64, batch_size: int = 32, shuffle: bool = False):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(root=image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def preprocess_image(img: Image.Image, image_size: int = 64) -> np.ndarray:
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    tensor = transform(img).unsqueeze(0)
    return tensor.numpy()