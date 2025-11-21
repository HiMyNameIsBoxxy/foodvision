import torch
from .food_embeddings import FOOD_FEATURES, FOOD_PROMPT_LIST

def predict_food_fast(emb, k=5):
    sims = emb @ FOOD_FEATURES.T
    vals, idxs = sims.topk(k)
    return [(FOOD_PROMPT_LIST[i], float(v)) for v, i in zip(vals, idxs)]

def clean_label(text):
    prefixes = ["a photo of ", "a dish of ", "a bowl of "]
    for p in prefixes:
        if text.startswith(p):
            return text[len(p):].strip()
    return text.strip()
