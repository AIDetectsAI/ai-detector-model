import asyncio
import io

from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image
from pydantic import BaseModel

from ai_detector_model.api.inference_controller import InferenceController
from ai_detector_model.api.model_schema import MODELS_BASE_PATH

app = FastAPI()

inference_engine = InferenceController(MODELS_BASE_PATH)


class CertaintyDTO(BaseModel):
    certainty: float
    model_used: str


@app.post("/verify/image", response_model=CertaintyDTO)
async def verify_image(file: UploadFile = File(...), type: str = Form(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    result = await asyncio.to_thread(inference_engine.predict, image)

    return CertaintyDTO(certainty=result, model_used=inference_engine.metadata.model_name)
