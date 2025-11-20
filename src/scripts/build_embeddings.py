import json
import torch
import open_clip
import numpy as np
from tqdm import tqdm

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"

def load_ingredients(path="data/ingredients_ontology.json"):
    with open(path, "r", encoding="utf-8") as f:
        ingredients = json.load(f)

    cleaned = []
    seen = set()
    for ing in ingredients:
        if isinstance(ing, str):
            ing = ing.strip().lower()
            if ing and ing not in seen:
                cleaned.append(ing)
                seen.add(ing)

    return cleaned

def main():
    ingredients = load_ingredients()
    print(f"Loaded {len(ingredients)} ingredients")

    prompts = [f"this dish contains {x}" for x in ingredients]

    # load model
    model, _, _ = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    BATCH = 512  # safe batch size for 8GB GPU
    text_embeds = []

    for i in tqdm(range(0, len(prompts), BATCH), desc="Encoding text"):
        batch_prompts = prompts[i:i + BATCH]

        tokens = tokenizer(batch_prompts).to(device)

        with torch.no_grad():
            features = model.encode_text(tokens)
            features /= features.norm(dim=-1, keepdim=True)

        text_embeds.append(features.cpu())

        # free VRAM
        del tokens, features
        torch.cuda.empty_cache()

    text_embeds = torch.cat(text_embeds, dim=0)  # (N, 512)

    print("Saving embeddings...")
    np.save("data/ingredient_features.npy", text_embeds.numpy())

    with open("data/ingredient_list.json", "w") as f:
        json.dump(ingredients, f, indent=2)

    print(f"Done. Saved {len(ingredients)} ingredient embeddings.")

if __name__ == "__main__":
    main()
