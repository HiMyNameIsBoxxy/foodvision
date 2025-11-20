import pandas as pd
import json
from pathlib import Path

DATA_DIR = Path("data/usda")

BRANDED = DATA_DIR / "branded_food.csv"
FOOD = DATA_DIR / "food.csv"
FOOD_NUTRIENT = DATA_DIR / "food_nutrient.csv"
NUTRIENT_ID_ENERGY = 1008  # replace with the ID you found if different

OUT_JSON = Path("data/calories_db.json")


def build_calories_db():
    print("Loading branded_food and food tables (metadata only)...")

    # 1) Load branded foods: fdc_id + descriptive fields
    branded = pd.read_csv(
        BRANDED,
        usecols=["fdc_id", "brand_name", "brand_owner", "subbrand_name",
                 "gtin_upc", "ingredients", "serving_size", "serving_size_unit"]
    )

    # 2) Load food table to get description/name
    food = pd.read_csv(
        FOOD,
        usecols=["fdc_id", "description"]
    )

    # Merge so we have nice names
    merged = branded.merge(food, on="fdc_id", how="left")

    # Build a human-readable name string
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
        return " ".join(name.split())  # collapse double spaces

    merged["pretty_name"] = merged.apply(build_name, axis=1)

    # Filter out empty names
    merged = merged[merged["pretty_name"].str.len() > 0]

    # We want a mapping: fdc_id -> pretty_name (+ maybe serving info)
    id_to_name = merged.set_index("fdc_id")["pretty_name"].to_dict()

    print(f"Branded foods with names: {len(id_to_name):,}")

    # 3) Now scan food_nutrient in chunks to pick out Energy rows only
    print("Scanning food_nutrient in chunks to extract Energy (kcal)...")
    calories_by_fdc = {}

    chunksize = 500_000
    for chunk_idx, chunk in enumerate(pd.read_csv(
        FOOD_NUTRIENT,
        chunksize=chunksize,
        usecols=["fdc_id", "nutrient_id", "amount"],  # amount is per 100g or per serving depending on data_type
    )):
        # Keep only Energy rows
        energy_rows = chunk[chunk["nutrient_id"] == NUTRIENT_ID_ENERGY]
        for _, row in energy_rows.iterrows():
            fdc_id = int(row["fdc_id"])
            amount = float(row["amount"])
            # If multiple entries for same fdc_id, keep the first or max; here we keep max just to be safe
            if fdc_id not in calories_by_fdc or amount > calories_by_fdc[fdc_id]:
                calories_by_fdc[fdc_id] = amount

        print(f"Processed chunk {chunk_idx}, total Energy rows so far: {len(calories_by_fdc):,}")

    print(f"Total foods with Energy: {len(calories_by_fdc):,}")

    # 4) Combine name + calories into a JSON
    calories_db = {}
    for fdc_id, kcal in calories_by_fdc.items():
        name = id_to_name.get(fdc_id)
        if not name:
            continue
        # Example key: "McDonald's Big Mac"
        calories_db[name] = float(kcal)

    print(f"Final entries in calories_db: {len(calories_db):,}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(calories_db, f, ensure_ascii=False, indent=2)

    print(f"Saved to {OUT_JSON}")


if __name__ == "__main__":
    build_calories_db()
