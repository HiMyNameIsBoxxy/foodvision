import os
import json
import math
import torch
import numpy as np

DATA_DIR = "data"

# Input files
CAL_JSON = os.path.join(DATA_DIR, "calories_simple.json")
FOOD_EMB_NPY = os.path.join(DATA_DIR, "food_features.npy")
FOOD_PROMPTS_JSON = os.path.join(DATA_DIR, "food_prompts.json")
ING_EMB_NPY = os.path.join(DATA_DIR, "ingredient_features.npy")
ING_LIST_JSON = os.path.join(DATA_DIR, "ingredient_list.json")

# Output files
OUT_CAL_PTH = os.path.join(DATA_DIR, "calories.pth")
OUT_FOOD_PTH = os.path.join(DATA_DIR, "food_embeds.pth")
OUT_ING_META = os.path.join(DATA_DIR, "ingredient_meta.pth")
OUT_CHUNK_DIR = os.path.join(DATA_DIR, "ingredient_chunks")

CHUNK_SIZE = 20000  # ~40–50MB chunks


def build_calorie_pth():
    print("Loading calorie DB...")
    with open(CAL_JSON, "r", encoding="utf-8") as f:
        cal = json.load(f)

    torch.save(cal, OUT_CAL_PTH)
    print(f"Saved → {OUT_CAL_PTH}")


def build_food_embeddings_pth():
    print("Loading food embeddings...")
    arr = np.load(FOOD_EMB_NPY)
    print("Food embed shape:", arr.shape)

    with open(FOOD_PROMPTS_JSON, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    torch.save(
        {"food_features": torch.tensor(arr, dtype=torch.float32),
         "food_prompts": prompts},
        OUT_FOOD_PTH,
    )
    print(f"Saved → {OUT_FOOD_PTH}")


def build_ingredient_chunks_pth():
    print("Loading ingredient embeddings...")
    arr = np.load(ING_EMB_NPY)
    total, dim = arr.shape

    print("Embedding shape:", arr.shape)

    with open(ING_LIST_JSON, "r", encoding="utf-8") as f:
        ing_list = json.load(f)

    os.makedirs(OUT_CHUNK_DIR, exist_ok=True)

    num_chunks = math.ceil(total / CHUNK_SIZE)
    print(f"Splitting into {num_chunks} chunks...")

    for i in range(num_chunks):
        start = i * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, total)

        chunk = torch.tensor(arr[start:end], dtype=torch.float32)

        out_path = os.path.join(OUT_CHUNK_DIR, f"ingredient_embeds_{i:02d}.pth")
        torch.save({"start": start, "end": end, "data": chunk}, out_path)

        print(f"Saved chunk {i} → {out_path} ({chunk.shape})")

    meta = {
        "total_rows": total,
        "dim": dim,
        "chunk_size": CHUNK_SIZE,
        "num_chunks": num_chunks,
        "ingredient_list": ing_list,
        "chunk_dir": OUT_CHUNK_DIR,
    }

    torch.save(meta, OUT_ING_META)
    print(f"Saved metadata → {OUT_ING_META}")


def main():
    print("=== BUILDING ALL PTH FILES ===")
    build_calorie_pth()
    build_food_embeddings_pth()
    build_ingredient_chunks_pth()
    print("\nAll .pth files built successfully ✔")


if __name__ == "__main__":
    main()
