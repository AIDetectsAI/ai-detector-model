import json
from pathlib import Path

import numpy as np
import onnxruntime
from PIL import Image

from ai_detector_model.api.model_factory import ensure_onnx_model
from ai_detector_model.api.model_schema import ModelMetadata
from ai_detector_model.data.image_preprocessor import preprocess_image


class InferenceController:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir

        with open(model_dir / "metadata.json") as f:
            self.metadata = ModelMetadata(**json.load(f))

        self.onnx_path = ensure_onnx_model(self.metadata, self.model_dir)

        self.ort_session = onnxruntime.InferenceSession(
            str(self.onnx_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.ort_session.get_inputs()[0].name

    def predict(self, image: Image.Image) -> float:
        preprocessed_image = preprocess_image(image, image_size=self.metadata.input_size)

        onnx_input = preprocessed_image.astype(np.float32)
        if not isinstance(onnx_input, np.ndarray):
            onnx_input = onnx_input.numpy().astype(np.float32)

        onnxruntime_output = self.ort_session.run(None, {self.input_name: onnx_input})[0]

        return float(onnxruntime_output.item())
