from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import Annotated, Any

from dotenv import load_dotenv
from loguru import logger
import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import efficientnet_b0
from tqdm import tqdm
import typer

app = typer.Typer(add_completion=False)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

load_dotenv()


def _safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_parse_class_map(raw_value: Any) -> dict[str, int]:
    if isinstance(raw_value, dict):
        return {str(k): int(v) for k, v in raw_value.items()}
    if isinstance(raw_value, str) and raw_value.strip():
        parsed = ast.literal_eval(raw_value)
        if isinstance(parsed, dict):
            return {str(k): int(v) for k, v in parsed.items()}
    return {}


def _load_metadata(model_path: Path) -> dict[str, Any]:
    if model_path.is_dir():
        metadata_path = model_path / "metadata.json"
    else:
        metadata_path = model_path.with_name("metadata.json")
    return _safe_read_json(metadata_path)


def _resolve_model_file(model_path: Path) -> Path:
    if model_path.is_dir():
        pth_path = model_path / "model.pth"
        if pth_path.exists():
            return pth_path
        raise FileNotFoundError(f"No model.pth found in {model_path}")
    return model_path


def _resolve_input_size(metadata: dict[str, Any]) -> int:
    if "input_size" in metadata:
        return int(metadata["input_size"])
    if "model" in metadata and isinstance(metadata["model"], dict):
        model_block = metadata["model"]
        if "input_size" in model_block:
            return int(model_block["input_size"])
    return 224


def _resolve_class_to_idx(dataset_root: Path, metadata: dict[str, Any]) -> dict[str, int]:
    class_map = _safe_parse_class_map(metadata.get("class_to_idx"))
    if class_map:
        return class_map
    return {"0_real": 0, "1_fake": 1}


def _build_transform(input_size: int):
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.ToTensor(),
        ]
    )


def _build_model(model_type: str) -> nn.Module:
    if model_type == "efficientnet":
        model = efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(
            in_features=model.classifier[1].in_features,
            out_features=1,
        )
        return model

    if model_type == "baseline_cnn":
        from models.pytorch.baseline_model_class import CustomBinaryCNN

        return CustomBinaryCNN()

    raise ValueError(f"Unsupported model_type: {model_type}")


class _FinalTestDataset(Dataset):
    def __init__(self, root: Path, input_size: int):
        self.root = root
        self.transform = _build_transform(input_size)
        self.samples: list[tuple[Path, int]] = []
        class_to_idx = {"0_real": 0, "1_fake": 1}

        for class_name, class_idx in class_to_idx.items():
            class_root = root / class_name
            if not class_root.exists():
                raise FileNotFoundError(f"Missing class directory: {class_root}")
            for path in sorted(class_root.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((path, class_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, label


def _load_torch_model(model_path: Path, metadata: dict[str, Any]) -> nn.Module:
    model_type = metadata.get("model_type")
    if not model_type and "model" in metadata and isinstance(metadata["model"], dict):
        model_type = metadata["model"].get("model_type")
    if not model_type:
        raise ValueError("Cannot infer model_type for PyTorch checkpoint")

    model = _build_model(str(model_type))
    loaded = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
    if isinstance(loaded, dict) and "state_dict" in loaded:
        loaded = loaded["state_dict"]
    if not isinstance(loaded, dict):
        raise ValueError(f"Unsupported checkpoint format: {model_path}")
    model.load_state_dict(loaded)
    model.eval()
    return model


def _resolve_device() -> torch.device:
    requested = os.getenv("TRAIN_DEVICE", "cpu").strip().lower()
    if requested == "cuda" and torch.cuda.is_available():
        logger.info("Using CUDA for evaluation")
        return torch.device("cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("TRAIN_DEVICE=cuda but CUDA is unavailable; falling back to CPU")
    else:
        logger.info("Using CPU for evaluation")
    return torch.device("cpu")


def _predict_with_torch(model: nn.Module, batch: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        output = model(batch).squeeze(-1)
    return output.detach().cpu().numpy()


def _compute_class_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, class_id: int
) -> dict[str, float]:
    binary_true = (y_true == class_id).astype(int)
    binary_pred = (y_pred == class_id).astype(int)
    binary_prob = y_prob if class_id == 1 else 1.0 - y_prob

    metrics = {
        "precision": precision_recall_fscore_support(
            binary_true, binary_pred, average="binary", zero_division=0
        )[0],
        "recall": precision_recall_fscore_support(
            binary_true, binary_pred, average="binary", zero_division=0
        )[1],
        "f1": f1_score(binary_true, binary_pred, zero_division=0),
        "accuracy": accuracy_score(binary_true, binary_pred),
        "mcc": matthews_corrcoef(binary_true, binary_pred),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(binary_true, binary_prob)
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return {key: float(value) for key, value in metrics.items()}


@app.command()
def main(
    model_path: Annotated[Path, typer.Argument(help="Path to model file or model directory")],
    final_test_dataset: Annotated[Path, typer.Argument(help="Path to final_test dataset root")],
    batch_size: Annotated[int, typer.Option(min=1, help="Inference batch size")] = 64,
    output_path: Annotated[
        Path | None,
        typer.Option(help="Optional JSON path for saving metrics"),
    ] = None,
) -> None:
    model_path = model_path.expanduser().resolve()
    final_test_dataset = final_test_dataset.expanduser().resolve()
    metadata = _load_metadata(model_path)
    model_file = _resolve_model_file(model_path)
    input_size = _resolve_input_size(metadata)
    class_to_idx = _resolve_class_to_idx(final_test_dataset, metadata)
    device = _resolve_device()

    logger.info(f"Model file: {model_file}")
    logger.info(f"Dataset root: {final_test_dataset}")
    logger.info(f"Input size: {input_size}")
    logger.info(f"Class mapping: {class_to_idx}")
    logger.info(f"Evaluation device: {device}")

    dataset = _FinalTestDataset(final_test_dataset, input_size=input_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )

    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[float] = []

    if model_file.suffix.lower() != ".pth":
        raise ValueError(f"Unsupported model file extension: {model_file.suffix}")

    model = _load_torch_model(model_file, metadata)
    model = model.to(device)
    backend = "torch"

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", total=len(loader)):
            images = images.to(device, non_blocking=True)
            outputs = _predict_with_torch(model, images)

            probs = 1.0 / (1.0 + np.exp(-outputs.reshape(-1)))
            preds = (probs >= 0.5).astype(int)

            y_true.extend(labels.numpy().tolist())
            y_pred.extend(preds.tolist())
            y_prob.extend(probs.tolist())

    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)
    y_prob_np = np.asarray(y_prob)

    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    class_metrics: dict[str, dict[str, float]] = {}
    for class_id, class_name in sorted(idx_to_class.items()):
        class_metrics[class_name] = _compute_class_metrics(
            y_true_np, y_pred_np, y_prob_np, class_id
        )

    macro_metrics = {
        "precision": float(np.mean([m["precision"] for m in class_metrics.values()])),
        "recall": float(np.mean([m["recall"] for m in class_metrics.values()])),
        "f1": float(np.mean([m["f1"] for m in class_metrics.values()])),
        "accuracy": float(accuracy_score(y_true_np, y_pred_np)),
        "mcc": float(matthews_corrcoef(y_true_np, y_pred_np)),
    }

    try:
        macro_metrics["roc_auc"] = float(roc_auc_score(y_true_np, y_prob_np))
    except ValueError:
        macro_metrics["roc_auc"] = float("nan")

    cm = confusion_matrix(y_true_np, y_pred_np, labels=sorted(idx_to_class))

    report = {
        "model_path": str(model_file),
        "dataset_path": str(final_test_dataset),
        "backend": backend,
        "input_size": input_size,
        "class_to_idx": class_to_idx,
        "macro": macro_metrics,
        "per_class": class_metrics,
        "confusion_matrix": cm.tolist(),
        "samples": len(y_true_np),
    }

    logger.success(json.dumps(report, indent=2, ensure_ascii=True))

    if output_path is not None:
        output_path = output_path.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
        logger.success(f"Saved report to {output_path}")


if __name__ == "__main__":
    app()
