# clip_multi_food.py (JSON-free + robust .pth loading)

import io
import os
import glob
import re
from typing import Dict, List
from PIL import Image
import torch
import open_clip
from rapidfuzz import process, fuzz

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------
# LOAD .PTH DATA FILES
# ------------------------------------------------------------
import glob
import os

# 1) Calories (dict: {name -> kcal})
CAL_DB = torch.load("data/calories.pth")
CAL_DB = {k: float(v) for k, v in CAL_DB.items()}

# 2) Food embeddings + prompts
food_blob = torch.load("data/food_embeds.pth", map_location=DEVICE)
FOOD_FEATURES = food_blob["food_features"].to(DEVICE)
FOOD_PROMPT_LIST = food_blob["food_prompts"]

# 3) Ingredient metadata
meta = torch.load("data/ingredient_meta.pth", map_location="cpu")

ING_LIST = meta["ingredient_list"]        # list of ingredient names
TOTAL_ING = meta["total_rows"]           # e.g. 249633
NUM_CHUNKS = meta["num_chunks"]          # 13

# 🚨 IMPORTANT FIX:
# Your files are NOT inside "data/ingredient_chunks"
# They are directly in "data/"
CHUNK_DIR = "data"


# ------------------------------------------------------------
# LOAD INGREDIENT EMBEDDING CHUNKS
# ------------------------------------------------------------
print(f"[INFO] Looking for ingredient chunks in: {CHUNK_DIR}")

# Match real files: ingredient_embeds_00.pth ... ingredient_embeds_12.pth
chunk_paths = sorted(glob.glob(os.path.join(CHUNK_DIR, "ingredient_embeds_*.pth")))

if len(chunk_paths) == 0:
    raise RuntimeError(f"No ingredient chunk .pth files found in: {CHUNK_DIR}")

if len(chunk_paths) != NUM_CHUNKS:
    print(f"[WARN] metadata says {NUM_CHUNKS} chunks but found {len(chunk_paths)} files")


ING_FEATURES_LIST = []

for path in chunk_paths:
    blob = torch.load(path, map_location="cpu")

    # Your build script uses: {"start":.., "end":.., "data": tensor}
    if isinstance(blob, dict) and "data" in blob:
        tensor = blob["data"]
    else:
        raise RuntimeError(f"Unexpected chunk format in {path}")

    ING_FEATURES_LIST.append(tensor)

# Merge all chunks
ING_FEATURES = torch.cat(ING_FEATURES_LIST, dim=0).to(DEVICE)

print(f"[OK] Loaded {TOTAL_ING} ingredient names")
print(f"[OK] Loaded {len(chunk_paths)} embedding chunks")
print(f"[OK] ING_FEATURES shape = {ING_FEATURES.shape}")



# ------------------------------------------------------------
# TEXT NORMALIZATION + FUZZY CALORIE MATCHING
# ------------------------------------------------------------

def _normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


NORM_CAL_DB = {_normalize_text(k): v for k, v in CAL_DB.items()}
CAL_KEYS = list(NORM_CAL_DB.keys())


def match_food_calories(food_label: str, threshold: int = 65):
    query = _normalize_text(food_label)
    if not query:
        return None, None, 0

    best = process.extractOne(query, CAL_KEYS, scorer=fuzz.token_set_ratio)
    if best is None:
        return None, None, 0

    key, score, _ = best
    if score < threshold:
        return None, None, score

    return key, NORM_CAL_DB[key], score


def token_fallback_match(food_label: str, min_score: int = 40):
    tokens = _normalize_text(food_label).split()

    best_key = None
    best_score = 0
    best_cal = None

    for t in tokens:
        best = process.extractOne(t, CAL_KEYS, scorer=fuzz.token_set_ratio)
        if best is None:
            continue

        key, score, _ = best
        if score > best_score:
            best_key = key
            best_score = score
            best_cal = NORM_CAL_DB[key]

    if best_score < min_score:
        return None, None, best_score

    return best_key, best_cal, best_score


# ------------------------------------------------------------
# CLIP MODEL
# ------------------------------------------------------------

_model = None
_preprocess = None
_tokenizer = None


def get_clip_model():
    global _model, _preprocess, _tokenizer

    if _model is None:
        model, preprocess, _ = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        tokenizer = open_clip.get_tokenizer("ViT-B-32")

        _model = model.to(DEVICE).eval()
        _preprocess = preprocess
        _tokenizer = tokenizer

        print("[OK] CLIP loaded.")

    return _model, _preprocess, _tokenizer


# ------------------------------------------------------------
# IMAGE ENCODING
# ------------------------------------------------------------

def encode_image(img: Image.Image) -> torch.Tensor:
    model, preprocess, tokenizer = get_clip_model()
    t = preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feat = model.encode_image(t)
        feat /= feat.norm(dim=-1, keepdim=True)

    return feat[0]


# ------------------------------------------------------------
# FOOD & INGREDIENT PREDICTIONS
# ------------------------------------------------------------

def predict_food_fast(emb: torch.Tensor, k: int = 5):
    sims = emb @ FOOD_FEATURES.T
    top_vals, top_idxs = sims.topk(k)

    results = []
    for s, idx in zip(top_vals, top_idxs):
        results.append((FOOD_PROMPT_LIST[int(idx)], float(s.item())))
    return results


def _clean_food_label(prompt: str) -> str:
    prefixes = [
        "a photo of ", "a close-up photo of ", "a professionally photographed ",
        "a plate of ", "a dish of ", "a top-down view of ",
        "a high-resolution photo of ", "a slice of ", "an italian-style ",
        "a baked ", "a bowl of ", "a steaming bowl of ",
        "a tray of ", "traditional japanese ",
    ]
    for p in prefixes:
        if prompt.startswith(p):
            return prompt[len(p):].strip()
    return prompt.strip()


def analyze_ingredients_fast(emb: torch.Tensor, k: int = 10):
    sims = emb @ ING_FEATURES.T
    top_vals, top_idxs = sims.topk(k)

    out = []
    for s, idx in zip(top_vals, top_idxs):
        out.append(
            {
                "name": ING_LIST[int(idx)],
                "score": float(s.item()),
            }
        )
    return out


# ------------------------------------------------------------
# MAIN API
# ------------------------------------------------------------

def analyze_food_image(image_bytes: bytes) -> Dict:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    emb = encode_image(image)

    # 1) Predict food label from CLIP
    food_res = predict_food_fast(emb)
    best_prompt, best_score = food_res[0]
    predicted_food = _clean_food_label(best_prompt)

    # LAYER 1 — Direct food-level calorie match
    mk, cal, ms = match_food_calories(predicted_food)
    if mk:
        return {
            "food": predicted_food,
            "confidence": best_score,
            "calorie_source": "food_match",
            "match_score": ms,
            "matched_key_norm": mk,
            "calories_per_100g": cal,
            "ingredients": [],
        }

    # LAYER 2 — Ingredient-based estimation
    ing_res = analyze_ingredients_fast(emb)
    seen = set()
    total_cal = 0.0
    details = []

    for pred in ing_res:
        raw = pred["name"]
        clip_score = pred["score"]

        mk, cal, ms = match_food_calories(raw, threshold=55)
        if mk is None or mk in seen:
            continue

        seen.add(mk)
        total_cal += cal

        details.append(
            {
                "ingredient_raw": raw,
                "ingredient_match_norm": mk,
                "calories_per_100g": cal,
                "ingredient_clip_score": clip_score,
                "match_score": ms,
            }
        )

    if total_cal > 0:
        return {
            "food": predicted_food,
            "confidence": best_score,
            "calorie_source": "ingredient_estimated",
            "calories_per_100g": total_cal,
            "ingredients": ing_res,
            "ingredient_calories_per_100g": details,
        }

    # LAYER 3 — Token fallback on food label
    fb_key, fb_cal, fb_score = token_fallback_match(predicted_food)
    return {
        "food": predicted_food,
        "confidence": best_score,
        "calorie_source": "fallback_token" if fb_key else "unknown",
        "matched_key_norm": fb_key,
        "match_score": fb_score,
        "calories_per_100g": fb_cal,
        "ingredients": ing_res,
    }
