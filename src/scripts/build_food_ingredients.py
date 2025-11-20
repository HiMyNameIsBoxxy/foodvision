import pandas as pd
import json
from pathlib import Path
import re

DATA_DIR = Path("data/usda")
BRANDED = DATA_DIR / "branded_food.csv"
FOOD = DATA_DIR / "food.csv"

OUT_FOOD_ING = Path("data/food_to_ingredients.json")
OUT_ING_VOCAB = Path("data/ingredients_ontology_from_usda.json")


def tokenize_ingredients(ing_str: str):
    if not isinstance(ing_str, str):
        return []

    # Basic clean
    s = ing_str.lower()

    # Remove parentheses contents (e.g., "wheat flour (enriched with ...)")
    s = re.sub(r"\([^)]*\)", " ", s)

    # Replace separators with commas
    s = s.replace(" and ", ",").replace(";", ",")

    # Split on commas
    parts = [p.strip() for p in s.split(",")]

    # Filter out very short or obviously junk tokens
    tokens = [p for p in parts if len(p) > 1]
    return tokens


def build_food_ingredients():
    print("Loading branded_food and food tables for ingredients...")

    branded = pd.read_csv(
        BRANDED,
        usecols=["fdc_id", "brand_owner", "brand_name", "subbrand_name", "ingredients"]
    )
    food = pd.read_csv(
        FOOD,
        usecols=["fdc_id", "description"]
    )

    merged = branded.merge(food, on="fdc_id", how="left")

    # Build pretty_name the same way as before
    def build_name(row):
        parts = []
        if pd.notna(row["brand_owner"]):
            parts.append(str(row["brand_owner"]).strip())
        if pd.notna(row["brand_name"]):
            parts.append(str(row["brand_name"]).strip())
        if pd.notna(row["subbrand_name"]):
            parts.append(str(row["subbrand_name"]).strip())
        if pd.notna(row["description"]):
            parts.append(str(row["description"]).strip())
        name = " ".join(parts)
        return " ".join(name.split())

    merged["pretty_name"] = merged.apply(build_name, axis=1)
    merged = merged[merged["pretty_name"].str.len() > 0]

    food_to_ingredients = {}
    vocab = set()

    print("Parsing ingredient strings...")
    for _, row in merged.iterrows():
        name = row["pretty_name"]
        ing_str = row["ingredients"]
        tokens = tokenize_ingredients(ing_str)
        if not tokens:
            continue
        food_to_ingredients[name] = tokens
        vocab.update(tokens)

    OUT_FOOD_ING.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FOOD_ING.open("w", encoding="utf-8") as f:
        json.dump(food_to_ingredients, f, ensure_ascii=False, indent=2)

    with OUT_ING_VOCAB.open("w", encoding="utf-8") as f:
        json.dump(sorted(vocab), f, ensure_ascii=False, indent=2)

    print(f"Saved food_to_ingredients.json with {len(food_to_ingredients):,} foods")
    print(f"Saved ingredients_ontology_from_usda.json with {len(vocab):,} unique tokens")


if __name__ == "__main__":
    build_food_ingredients()
