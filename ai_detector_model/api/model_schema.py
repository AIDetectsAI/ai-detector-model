import ast
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class ModelMetadata(BaseModel):
    model_name: str
    model_type: str
    input_size: int
    class_to_idx: str | dict

    @property
    def parsed_classes(self) -> dict:
        if isinstance(self.class_to_idx, str):
            return ast.literal_eval(self.class_to_idx)
        return self.class_to_idx


ACTIVE_MODEL_NAME = os.getenv("ACTIVE_MODEL_DIR", "latent-stable-model")

MODELS_PYTORCH_BASE_PATH = Path("models") / "pytorch" / ACTIVE_MODEL_NAME
MODELS_ONNX_BASE_PATH = Path("models") / "onnx" / ACTIVE_MODEL_NAME
