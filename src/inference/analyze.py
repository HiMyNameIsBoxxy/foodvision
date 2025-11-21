from PIL import Image
import io
from .clip_model import encode_image
from .food_predictor import predict_food_fast, clean_label
from .ingredient_predictor import analyze_ingredients_fast
from .calorie_matcher import match_food_calories, token_fallback_match

def analyze_food_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    emb = encode_image(img)

    top_label, conf = predict_food_fast(emb)[0]
    food = clean_label(top_label)

    mk, cal, ms = match_food_calories(food)
    ingredients = analyze_ingredients_fast(emb)

    if mk:
        return {
            "food": food,
            "confidence": conf,
            "calorie_source": "food_match",
            "matched_key_norm": mk,
            "match_score": ms,
            "calories_per_100g": cal,
            "ingredients": ingredients,
        }

    # fallback...
    fb_key, fb_cal, fb_score = token_fallback_match(food)
    return {
        "food": food,
        "confidence": conf,
        "calorie_source": "fallback_token",
        "matched_key_norm": fb_key,
        "match_score": fb_score,
        "calories_per_100g": fb_cal,
        "ingredients": ingredients,
    }
