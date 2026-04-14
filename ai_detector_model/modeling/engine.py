import os.path
from typing import Any

from loguru import logger
import mlflow
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
        output_dir: str,
        use_amp: bool,
        metric_name: str = "f1",
        metric_mode: str = "max",
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optim = optim
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.output_dir = output_dir
        self.use_amp = use_amp
        self.metric_name = metric_name
        self.metric_mode = metric_mode

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
        for i, (images, labels) in enumerate(self.train_loader):
            self.optim.zero_grad()
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.autocast(
                device_type=self.device, dtype=torch.bfloat16, enabled=self.use_amp
            ):
                pred = self.model(images).squeeze(-1)
                loss = self.loss_fn(pred, labels.float())

            losses.append(loss.item())
            loss.backward()
            self.optim.step()

            if i % 100 == 0:
                logger.info(f"Epoch {idx}, Batch {i}/{n}")

        epoch_loss = np.mean(losses)
        mlflow.log_metric('train_loss_epoch', epoch_loss, step=idx)

        current_lr = self.optim.param_groups[0]['lr']
        mlflow.log_metric("lr", current_lr, step=idx)
        
        return epoch_loss

    def evaluate(self, epoch_idx: int) -> dict:
        if self.test_loader is None:
            raise ValueError("TestLoader cannot be None")
        self.model.eval()
        self.metrics.reset()

        images: Tensor
        labels: Tensor
        losses = []
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with torch.autocast(
                    device_type=self.device, dtype=torch.bfloat16, enabled=self.use_amp
                ):
                    pred = self.model(images).squeeze(-1)
                    loss = self.loss_fn(pred, labels.float())

                losses.append(loss.item())

                self.metrics.update(torch.sigmoid(pred.float()), labels)

        scores = {k: v.item() for k, v in self.metrics.compute().items()}
        scores["validation_loss"] = np.mean(losses)

        for metric_name, value in scores.items():
            mlflow.log_metric(f"val_{metric_name}", value, step=epoch_idx)

        return scores

    def train(self, max_epochs: int):
        if self.train_loader is None or self.test_loader is None:
            raise ValueError("DataLoaders cannot be None")
        best_metric = -float("inf") if self.metric_mode == "max" else float("inf")
        best_epoch = -1
        model_path = os.path.join(self.output_dir, "model.pth")
        for epoch in range(max_epochs):
            logger.info(f"Epoch: {epoch}")
            train_loss = self.train_one_epoch(epoch)
            scores = self.evaluate(epoch)

            scores["training_loss"] = train_loss
            logger.info(f"Metrics for epoch {epoch}: {scores}")

            if self.metric_name not in scores:
                raise ValueError(
                    f"Metric '{self.metric_name}' not found! "
                    f"Available metrics are: {list(scores.keys())}"
                )

            curr_metric = scores[self.metric_name]

            if (
                curr_metric > best_metric
                if self.metric_mode == "max"
                else curr_metric < best_metric
            ):
                logger.info(
                    f"New model with better {self.metric_name} found: "
                    f"{self.metric_name} = {curr_metric}"
                )
                best_metric = curr_metric
                best_epoch = epoch
                torch.save(self.model.state_dict(), model_path)
                #mlflow.pytorch.log_model(self.model, artifact_path='model_best',
                # log_datasets=False)

        logger.info("TRAINING FINISHED")
        logger.info(f"Best {self.metric_name} achieved: {best_metric} in epoch {best_epoch}")
