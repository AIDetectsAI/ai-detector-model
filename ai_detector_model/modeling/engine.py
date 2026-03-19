import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torchmetrics
from torchmetrics import Accuracy, F1Score, Precision, Recall


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optim: Optimizer,
        train_loader: DataLoader | None,
        test_loader: DataLoader | None,
        device: str,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optim = optim
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

        self.metrics = torchmetrics.MetricCollection(
            {
                "accuracy": Accuracy(task="binary"),
                "precision": Precision(task="binary"),
                "recall": Recall(
                    task="binary",
                ),
                "f1": F1Score(task="binary"),
            }
        ).to(device=self.device)

    def train_one_epoch(self, idx: int):
        pass

    def evaluate(self):
        self.model.eval()

    def train(self):
        self.model.train()
