from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from ai_detector_model.config import *
from ai_detector_model.model_converter import ModelController
import asyncio
import io
from PIL import Image
from ai_detector_model.image_preprocessor import preprocess_image

app = FastAPI()

class APIController():
    def __init__(self):
        self.model_controller = ModelController("models/onnx/baseline_model.onnx")

    async def get_image_certainty(self, file: File, type: str) -> float:
        contents = file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        preprocessed_image = preprocess_image(image)
        result = await asyncio.to_thread(self.model_controller.run_onnx_model, preprocessed_image)
        return result
   
model_handler = APIController()

class CertaintyDTO(BaseModel):
    certainty: float

@app.post("/verify/image")
async def verify_image(file: UploadFile = File(...), type: str = Form(...)):
    certainty = await model_handler.get_image_certainty(file, type)
    return CertaintyDTO(certainty=certainty)