from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from src.inference.clip_multi_food import analyze_food_image

app = FastAPI(title="FoodVision CLIP API")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = analyze_food_image(image_bytes)
    return JSONResponse(result)
