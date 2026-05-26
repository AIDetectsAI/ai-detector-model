import asyncio
import io
from typing import cast

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from ai_detector_model.api.inference_controller import InferenceController
from ai_detector_model.api.model_factory import get_pytorch_model_dir
from ai_detector_model.api.model_schema import ACTIVE_MODEL_NAME

app = FastAPI()

inference_engine = None


def get_inference_engine() -> InferenceController:
    global inference_engine
    if inference_engine is None:
        inference_engine = InferenceController(get_pytorch_model_dir(ACTIVE_MODEL_NAME))
    return cast(InferenceController, inference_engine)


class CertaintyDTO(BaseModel):
    certainty: float
    # model_used: str
    # class_to_idx: dict[str, int]


@app.post("/verify/image", response_model=CertaintyDTO)
async def verify_image(file: UploadFile = File(...), type: str = Form(...)):
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Provided file was not an image") from None

    engine = get_inference_engine()
    result = await asyncio.to_thread(engine.predict, image)

    return CertaintyDTO(
        certainty=result,
        # model_used=inference_engine.metadata.model_name,
        # class_to_idx=inference_engine.class_to_idx,
    )
