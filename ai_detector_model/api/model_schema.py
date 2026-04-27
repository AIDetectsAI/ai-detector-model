import ast
import os
from pathlib import Path

from pydantic import BaseModel


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


ACTIVE_MODEL_DIR = os.getenv("ACTIVE_MODEL_DIR", "latent-stable-model_v1")
MODELS_BASE_PATH = Path("models") / ACTIVE_MODEL_DIR
