import asyncio
import io
from typing import cast

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from ai_detector_model.api.caption_controller import CaptionController
from ai_detector_model.api.inference_controller import InferenceController
from ai_detector_model.api.model_factory import get_pytorch_model_dir
from ai_detector_model.api.model_schema import ACTIVE_MODEL_NAME

app = FastAPI()

inference_engine = None
caption_engine = None


def get_inference_engine() -> InferenceController:
    global inference_engine
    if inference_engine is None:
        inference_engine = InferenceController(get_pytorch_model_dir(ACTIVE_MODEL_NAME))
    return cast(InferenceController, inference_engine)


def get_caption_engine() -> CaptionController:
    global caption_engine
    if caption_engine is None:
        caption_engine = CaptionController()
    return cast(CaptionController, caption_engine)


class CertaintyDTO(BaseModel):
    certainty: float
    caption: str
    # model_used: str
    # class_to_idx: dict[str, int]


@app.post("/verify/image", response_model=CertaintyDTO)
async def verify_image(file: UploadFile = File(...), type: str = Form(...)):
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Provided file was not an image") from None

    inf_engine = get_inference_engine()
    cap_engine = get_caption_engine()

    task_verify = asyncio.to_thread(inf_engine.predict, image)
    task_caption = asyncio.to_thread(cap_engine.generate_caption, image)

    certainty, caption = await asyncio.gather(task_verify, task_caption)

    return CertaintyDTO(
        certainty=certainty,
        caption=caption,
        # model_used=inference_engine.metadata.model_name,
        # class_to_idx=inference_engine.class_to_idx,
    )
