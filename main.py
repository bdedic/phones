from fastapi import FastAPI, UploadFile, File, Query
from app.utils import get_embedding
import pickle
import torch
import torch.nn.functional as F

app = FastAPI()

# Load reference embeddings: dict[label] = tensor[num_images, emb_dim]
with open("reference_embeddings_all.pkl", "rb") as f:
    reference_data = pickle.load(f)

# Flatten reference data for fast searching
all_labels = []
all_embeddings = []

for label, embs in reference_data.items():
    all_labels.extend([label] * embs.size(0))
    all_embeddings.append(embs)

all_embeddings = torch.cat(all_embeddings, dim=0)  # Shape: (N, emb_dim)

@app.get("/")
async def root():
    return {"status": "API is running"}

@app.post("/predict/")
async def predict(
    file: UploadFile = File(...),
    threshold: float = Query(0.3, ge=0.0, le=1.0),
    k: int = Query(3, ge=1, le=10)
):
    contents = await file.read()
    print(f"Received image: {file.filename} ({len(contents)} bytes)")

    temp_path = "temp.jpg"
    with open(temp_path, "wb") as f:
        f.write(contents)

    embedding = get_embedding(temp_path)
    if embedding is None or not isinstance(embedding, torch.Tensor):
        print("get_embedding() returned None or invalid format.")
        return {"model": None, "confidence": 0.0, "alternatives": []}

    print("Embedding shape:", embedding.shape)
    embedding = F.normalize(embedding, p=2, dim=0)

    # Compute cosine similarities
    similarities = F.cosine_similarity(embedding.unsqueeze(0), all_embeddings)
    topk_sim, topk_idx = torch.topk(similarities, k)

    topk_labels = [all_labels[i] for i in topk_idx]
    topk_sims = topk_sim.tolist()
    print("Top predictions:")
    for label, score in zip(topk_labels, topk_sims):
        print(f" - {label}: {score:.4f}")

    # Weighted voting
    votes = {}
    for label, sim in zip(topk_labels, topk_sim):
        votes[label] = votes.get(label, 0.0) + sim.item()

    if not votes:
        print("No votes accumulated.")
        return {"model": None, "confidence": 0.0, "alternatives": []}

    best_label = max(votes, key=votes.get)
    best_score = votes[best_label] / k
    print(f"Best label: {best_label} | Score: {best_score:.4f}")

    if best_score < threshold:
        print("Below threshold. Returning null.")
        return {
            "model": None,
            "confidence": round(best_score, 4),
            "alternatives": [
                {"label": l, "score": round(s, 4)} for l, s in zip(topk_labels, topk_sims)
            ]
        }

    return {
        "model": best_label,
        "confidence": round(best_score, 4),
        "alternatives": [
            {"label": l, "score": round(s, 4)} for l, s in zip(topk_labels, topk_sims)
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000, reload=True)
