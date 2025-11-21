import open_clip, torch
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model = None
_preprocess = None
_tokenizer = None

def get_clip_model():
    global _model, _preprocess, _tokenizer
    if _model is None:
        _model, _preprocess, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        _tokenizer = open_clip.get_tokenizer("ViT-B-32")
        _model = _model.to(DEVICE).eval()
    return _model, _preprocess, _tokenizer

def encode_image(image: Image.Image):
    model, preprocess, _ = get_clip_model()
    t = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = model.encode_image(t)
        feat /= feat.norm(dim=-1, keepdim=True)
    return feat[0]
