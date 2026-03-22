from typing import Any

from loguru import logger
import numpy as np
from numpy import floating
import torch
from torch import Tensor
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

    def train_one_epoch(self, idx: int) -> floating[Any]:
        self.model.train()
        n = len(self.train_loader)

        losses = []
        images: Tensor
        labels: Tensor
        self.optim.zero_grad()
        for i, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            pred = self.model(images).squeeze(-1)
            loss = self.loss_fn(pred, labels.float())
            loss.backward()
            losses.append(loss.item())

            self.optim.step()
            self.optim.zero_grad()

            if i % 100 == 0:
                logger.info(f"Epoch {idx}, Batch {i}/{n}")

        return np.mean(losses)

    def evaluate(self) -> dict:
        self.model.eval()
        self.metrics.reset()

        images: Tensor
        labels: Tensor
        losses = []
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                pred = self.model(images).squeeze(-1)

                loss = self.loss_fn(pred, labels.float())
                losses.append(loss.item())

                self.metrics.update(torch.sigmoid(pred), labels)

        scores = {k: v.item() for k, v in self.metrics.compute().items()}
        scores["validation_loss"] = np.mean(losses)

        return scores

    def train(self, max_epochs: int):
        best_f1 = 0.0
        best_epoch = -1
        model_path = "curr_best_model.pth"
        for epoch in range(max_epochs):
            logger.info(f"Epoch: {epoch}")
            train_loss = self.train_one_epoch(epoch)
            scores = self.evaluate()

            scores["training_loss"] = train_loss
            logger.info(f"Metrics for epoch {epoch}: {scores}")

            curr_f1 = scores["f1"]
            if curr_f1 > best_f1:
                logger.info(f"New model with better F1 found: f1 = {curr_f1}")
                best_f1 = curr_f1
                best_epoch = epoch
                torch.save(self.model.state_dict(), model_path)

        logger.info("TRAINING FINISHED")
        logger.info(f"Best f1 score achieved: {best_f1} in epoch {best_epoch}")
