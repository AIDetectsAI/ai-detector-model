import cv2
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.base_dataset = ImageFolder(root=root_dir)
        self.classes = self.base_dataset.classes
        self.targets = self.base_dataset.targets
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        path, label = self.base_dataset.samples[idx]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label
