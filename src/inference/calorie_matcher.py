import re
import torch
from rapidfuzz import process, fuzz

# Load the calorie database
CAL_DB = torch.load("data/calories.pth")
CAL_DB = {k: float(v) for k, v in CAL_DB.items()}

def _normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Normalized calorie DB + keys
NORM_CAL_DB = {_normalize_text(k): v for k, v in CAL_DB.items()}
CAL_KEYS = list(NORM_CAL_DB.keys())


# xact / fuzzy food match
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



# Token fallback match
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
