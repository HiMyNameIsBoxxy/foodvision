# scripts/build_food_embeddings.py

import json
import torch
import open_clip
import numpy as np
from tqdm import tqdm


MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"


def load_food_list(path="data/food_list_2000.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    foods = []
    for _cuisine, cats in data.items():
        for _cat, items in cats.items():
            for item in items:
                if isinstance(item, str):
                    foods.append(item.strip())

    # ◼️ Prompt expansion (same forms you use during prediction)
    prompts = []
    for f in foods:
        f_clean = f.strip()

        prompts.extend([
            f"a photo of {f_clean}",
            f"a close-up photo of {f_clean}",
            f"a professionally photographed {f_clean}",
            f"a plate of {f_clean}",
            f"a dish of {f_clean}",
            f"a top-down view of {f_clean}",
            f"a high-resolution photo of {f_clean}",
        ])

        if "pizza" in f_clean:
            prompts.extend([
                f"a slice of {f_clean}",
                f"an Italian-style {f_clean}",
                f"a baked {f_clean} on a plate",
            ])

        if any(x in f_clean for x in ["ramen", "pasta", "noodles", "spaghetti"]):
            prompts.extend([
                f"a bowl of {f_clean}",
                f"a steaming bowl of {f_clean}",
            ])

        if "sushi" in f_clean:
            prompts.extend([
                f"a tray of {f_clean}",
                f"traditional Japanese {f_clean}",
            ])

    return foods, prompts


def main():
    foods, prompts = load_food_list()
    print(f"Loaded {len(foods)} food items → {len(prompts)} text prompts")

    # Load CLIP
    model, _, _ = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    BATCH = 512  # safe for 8GB GPU
    all_features = []

    for i in tqdm(range(0, len(prompts), BATCH), desc="Encoding food prompts"):
        batch_prompts = prompts[i:i + BATCH]

        tokens = tokenizer(batch_prompts).to(device)

        with torch.no_grad():
            features = model.encode_text(tokens)
            features /= features.norm(dim=-1, keepdim=True)

        all_features.append(features.cpu())

        del tokens, features
        torch.cuda.empty_cache()

    all_features = torch.cat(all_features, dim=0)  # (N,512)

    # Save
    print("Saving...")
    np.save("data/food_features.npy", all_features.numpy())
    with open("data/food_prompts.json", "w") as f:
        json.dump(prompts, f, indent=2)

    print(f"Saved {len(prompts)} food prompt embeddings.")


if __name__ == "__main__":
    main()
