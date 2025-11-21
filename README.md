# FoodVision — CLIP‑Powered Food & Calorie Estimation (FastAPI + Docker)

This project provides a **production‑grade FastAPI service** that performs:

- **Food recognition** 
- **Ingredient recognition**  
- **Calorie estimation** 
- All embeddings stored in`.pth` chunks  
- Fully packaged and deployable with **Docker**
- No internet connection required at runtime

---

## Features

### Food Recognition (CLIP)
- Uses **OpenAI CLIP ViT‑B/32** model  
- Predicts food labels from precomputed prompt embeddings  
- Returns top‑k foods with confidence scores  
- Provides clean, human‑friendly labels (strips prefixes like *"a photo of ..."*)

### Ingredient Recognition
- Over **249,000** ingredient embeddings  
- Stored efficiently in **13 x ~40MB chunks**  
- Fast cosine similarity search using preloaded tensors  
- Returns ingredient names + similarity scores

### Calorie Estimation
Supports 3‑layer calorie inference:

1. **Direct food‑level match**  
2. **Ingredient‑based calorie estimation**  
3. **Token fallback match**

Uses **fuzzy string matching** (RapidFuzz) over normalized calorie keys.

---


## Run With Docker

### 1. Build the Docker image
```bash
docker build -t foodvision .
```

### 2. Run the container
```bash
docker run -p 8000:8000 foodvision
```

### 3. Open API docs
Visit:

```
http://127.0.0.1:8000/docs
```

---

## API Endpoints

### **POST /analyze**
Analyze an uploaded food image:

**Response Includes:**
- Predicted food name  
- CLIP confidence  
- Calorie source  
- Ingredient list  
- Calorie estimate  

---

## Dependencies

- Python 3.10+
- PyTorch  
- FastAPI  
- Uvicorn  
- OpenCLIP  
- RapidFuzz  
- Pillow  
- NumPy  

Installed automatically inside Docker.

---
