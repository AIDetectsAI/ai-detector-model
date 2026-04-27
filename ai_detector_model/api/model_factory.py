from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

from ai_detector_model.api.model_schema import ModelMetadata


class ModelFactory:
    @staticmethod
    def get_pytorch_model(model_name: str, num_classes: int):
        if model_name == "EFFICIENTNET-B0":
            model = efficientnet_b0()
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            return model
        else:
            raise ValueError(f"Not supported model: {model_name}")


def ensure_onnx_model(metadata: ModelMetadata, model_dir: Path) -> Path:
    onnx_path = model_dir / "model.onnx"
    pth_path = model_dir / "model.pth"

    if onnx_path.exists():
        return onnx_path

    print(f"No onnx file. Starting conversion... {metadata.model_type}...")
    num_classes = len(metadata.parsed_classes)

    model = ModelFactory.get_pytorch_model(metadata.model_type, num_classes)
    model.load_state_dict(torch.load(pth_path, map_location=torch.device("cpu")))
    model.eval()

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
