import re
from unidecode import unidecode
import spacy

# Load a FAST spaCy pipeline (no parser, no NER)
# This loads instantly and runs 20–30× faster
try:
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "tagger"])
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "tagger"])

BLOCKLIST = [
    "company", "limited", "inc", "co", "brand",
    "products", "product", "market", "llc", "ltd",
    "organic", "premium", "homemade", "seasoning",
]

ALLOWED_FOODS = [
    "broth", "oil", "soup", "cream", "flour",
    "tomato", "beef", "chicken", "clam", "broccoli",
    "vegetable", "basil", "ham", "noodle", "rice",
    "bean", "pepper", "pasta", "cheese"
]

def clean_usda_name(raw: str) -> str | None:
    """Very fast cleaning using tok2vec SpaCy + regex pattern rules."""
    if not raw:
        return None

    text = unidecode(raw).lower()

    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\b[0-9]+\s?(oz|g|ml|lb|gal|kg)\b", " ", text)
    text = re.sub(r"[\W_]+", " ", text)

    for bad in BLOCKLIST:
        text = re.sub(rf"\b{bad}\b", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return None

    doc = nlp(text)

    tokens = []
    for t in doc:
        if t.text in ALLOWED_FOODS or t.pos_ in ["NOUN"]:
            tokens.append(t.lemma_)

    if not tokens:
        return None

    # dedupe
    seen = set()
    cleaned = []
    for t in tokens:
        if t not in seen:
            cleaned.append(t)
            seen.add(t)

    return " ".join(cleaned)
