from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from .model_handler import ModelController
app = FastAPI()

class APIController():
    def __init__(self):
        self.model_controller = ModelController("models/onnx/baseline_model.onnx")

    async def get_image_certainty(self, file: File, type: str) -> float:
        preprocessed_image = 1 # convert to proper format
        return self.model_controller.run_onnx_model(preprocessed_image)

model_handler = APIController()

class CertaintyDTO(BaseModel):
    certainty: float

@app.post("/verify/image")
async def verify_image(file: UploadFile = File(...), type: str = Form(...)):
    certainty = await model_handler.get_image_certainty(file, type)
    return {"certainty": certainty}