from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel

app = FastAPI()

class ModelController():
    def __init__(self):
        # TODO initialize model
        print("Initializing...")

    async def get_image_certainty(file: File, type: str) -> float:
        # TODO call model for certainty
        return 0.69

model_handler = ModelController()

class CertaintyDTO(BaseModel):
    certainty: float

@app.post("/verify/image")
async def verify_image(file: UploadFile = File(...), type: str = Form(...)):
    certainty = await model_handler.get_image_certainty(file, type)
    return {"certainty": certainty}