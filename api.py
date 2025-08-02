from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel

# for future removal
import random

app = FastAPI()

class CertaintyDTO(BaseModel):
    certainty: float

@app.post("/verify")
async def verity_image(file: UploadFile = File(...), type: str = Form(...)):
    certainty = await get_certainty(file, type)
    return {"certainty": certainty}

async def get_certainty(file, type):
    # TODO / temporary placeholder return value
    return random.randrange(1,100)/100