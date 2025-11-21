import torch
from .ingredient_embeddings import ING_FEATURES, ING_LIST

def analyze_ingredients_fast(emb, k=10):
    sims = emb @ ING_FEATURES.T
    vals, idxs = sims.topk(k)

    return [
        {"name": ING_LIST[i], "score": float(v)}
        for v, i in zip(vals, idxs)
    ]
