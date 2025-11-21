import torch, glob, os

meta = torch.load("data/ingredient_meta.pth", map_location="cpu")
ING_LIST = meta["ingredient_list"]
NUM_CHUNKS = meta["num_chunks"]

chunk_paths = sorted(glob.glob("data/ingredient_embeds_*.pth"))
ING_FEATURES = torch.cat(
    [torch.load(p, map_location="cpu")["data"] for p in chunk_paths],
    dim=0
).to("cuda" if torch.cuda.is_available() else "cpu")
