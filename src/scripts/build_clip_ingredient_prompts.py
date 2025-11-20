import json
from pathlib import Path

IN_ING = Path("data/ingredients_ontology.json")
OUT_PROMPTS = Path("data/ingredient_prompts.json")


def build_prompts(ingredients):
    prompts = []
    for ing in ingredients:
        prompts.extend([
            f"a dish containing {ing}",
            f"{ing} visible in the food",
            f"a close-up photo showing {ing}",
            f"{ing} on top of the dish",
            f"this food contains {ing}"
        ])
    return prompts


if __name__ == "__main__":
    ingredients = json.load(open(IN_ING, "r", encoding="utf-8"))
    prompts = build_prompts(ingredients)
    json.dump(prompts, open(OUT_PROMPTS, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"Generated {len(prompts):,} ingredient prompts.")
