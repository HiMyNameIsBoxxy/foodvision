import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from src.inference.analyze import analyze_food_image

app = FastAPI(title="FoodVision CLIP API")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image bytes
    image_bytes = await file.read()

    # Run CLIP
    result = analyze_food_image(image_bytes)

    # Add base64 image so frontend can render it
    encoded = base64.b64encode(image_bytes).decode("utf-8")

    # Final JSON response
    response = {
        "prediction": result,
        "image_base64": encoded
    }

    return JSONResponse(response)
