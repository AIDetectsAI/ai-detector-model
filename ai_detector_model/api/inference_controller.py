import json
import math
from pathlib import Path

import numpy as np
import onnxruntime
from PIL import Image

from ai_detector_model.api.model_factory import ensure_onnx_model, get_onnx_model_dir
from ai_detector_model.api.model_schema import ModelMetadata
from ai_detector_model.data.image_preprocessor import preprocess_image


class InferenceController:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.model_name = model_dir.name

        with open(model_dir / "metadata.json") as f:
            self.metadata = ModelMetadata(**json.load(f))

        self.onnx_path = ensure_onnx_model(self.metadata, self.model_dir)
        self.onnx_dir = get_onnx_model_dir(self.metadata.model_name)

        self.ort_session = onnxruntime.InferenceSession(
            str(self.onnx_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.ort_session.get_inputs()[0].name

    @property
    def class_to_idx(self) -> dict[str, int]:
        return self.metadata.parsed_classes

    def predict(self, image: Image.Image) -> float:
        preprocessed_image = preprocess_image(image, image_size=self.metadata.input_size)

        onnx_input = preprocessed_image.astype(np.float32)
        if not isinstance(onnx_input, np.ndarray):
            onnx_input = onnx_input.numpy().astype(np.float32)

        onnxruntime_output = self.ort_session.run(None, {self.input_name: onnx_input})[0]

        logits = float(np.asarray(onnxruntime_output).reshape(-1)[0])
        certainty = 1.0 / (1.0 + math.exp(-logits))

        return float(certainty)
