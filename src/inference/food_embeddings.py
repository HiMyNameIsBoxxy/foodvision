import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

blob = torch.load("data/food_embeds.pth", map_location=DEVICE)
FOOD_FEATURES = blob["food_features"].to(DEVICE)
FOOD_PROMPT_LIST = blob["food_prompts"]
