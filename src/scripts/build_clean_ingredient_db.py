import json
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from nlp_cleaner import clean_usda_name

RAW_DB_PATH = Path("data/calories_db.json")
OUTPUT_PATH = Path("data/clean_ingredient_calories.json")

# ⚡ Best batch size
BATCH_SIZE = 5000

def process_batch(items):
    """Process a batch of (key, cal) pairs and return cleaned list."""
    results = []
    for raw_name, cal in items:
        cleaned = clean_usda_name(raw_name)
        if cleaned is None:
            continue

        if len(cleaned.split()) == 1 and cleaned not in ["beef", "chicken", "tomato", "oil"]:
            continue

        results.append((cleaned, float(cal)))
    return results


def build_clean_db():
    with open(RAW_DB_PATH, "r", encoding="utf-8") as f:
        raw_db = list(json.load(f).items())

    total = len(raw_db)
    print(f"Loaded {total} USDA items.")

    # Split into batches
    batches = [
        raw_db[i:i + BATCH_SIZE]
        for i in range(0, total, BATCH_SIZE)
    ]

    print(f"Processing in {len(batches)} batches with {cpu_count()} workers...\n")

    cleaned_dict = {}

    with Pool(cpu_count()) as pool:
        for batch_results in tqdm(pool.imap(process_batch, batches), total=len(batches)):
            for name, calories in batch_results:
                if name not in cleaned_dict:
                    cleaned_dict[name] = calories

    print("\n===== DONE =====")
    print(f"Cleaned ingredient entries: {len(cleaned_dict)}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(cleaned_dict, f, indent=2)

    print(f"Saved cleaned DB to: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_clean_db()
