from pathlib import Path
import re

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, efficientnet_b1

from ai_detector_model.api.model_schema import ModelMetadata


class ModelFactory:
    @staticmethod
    def get_pytorch_model(model_name: str, num_classes: int):
        if model_name == "EFFICIENTNET-B0":
            model = efficientnet_b0(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes - 1)
            return model
        elif model_name == "EFFICIENTNET-B1":
            model = efficientnet_b1(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes - 1)
            return model
        else:
            raise ValueError(f"Not supported model: {model_name}")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def get_pytorch_model_dir(model_name: str) -> Path:
    return Path("models") / "pytorch" / _safe_name(model_name)


def get_onnx_model_dir(model_name: str) -> Path:
    return Path("models") / "onnx" / _safe_name(model_name)


def ensure_onnx_model(metadata: ModelMetadata, pytorch_model_dir: Path) -> Path:
    onnx_dir = get_onnx_model_dir(pytorch_model_dir.name)
    onnx_path = onnx_dir / "model.onnx"
    pth_path = pytorch_model_dir / "model.pth"

    if onnx_path.exists():
        return onnx_path

    print(f"No onnx file. Starting conversion for {pytorch_model_dir.name}...")
    num_classes = len(metadata.parsed_classes)

    model = ModelFactory.get_pytorch_model(metadata.model_name, num_classes)
    model.load_state_dict(torch.load(pth_path, map_location=torch.device("cpu")))
    model.eval()

    onnx_dir.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(1, 3, metadata.input_size, metadata.input_size)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )
    print("Conversion successfull.")
    return onnx_path
