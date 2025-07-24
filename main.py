from fastapi import FastAPI, UploadFile, File
from app.utils import get_embedding
import pickle
import torch
import torch.nn.functional as F
import io

app = FastAPI()

# Load your reference embeddings: dict[label] = tensor[num_images, emb_dim]
with open("reference_embeddings_all.pkl", "rb") as f:
    reference_data = pickle.load(f)

# Prepare flat lists for efficient searching
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
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    temp_path = "temp.jpg"
    with open(temp_path, "wb") as f:
        f.write(contents)

    # Get embedding
    embedding = get_embedding(temp_path)
    if embedding is None or not isinstance(embedding, torch.Tensor):
        print("❌ Invalid embedding returned")
        return {"model": None, "confidence": 0.0}

    embedding = F.normalize(embedding, p=2, dim=0)
    print("✅ Embedding shape:", embedding.shape)

    similarities = F.cosine_similarity(embedding.unsqueeze(0), all_embeddings)
    print("📊 Top 10 similarities:", similarities.topk(10).values.tolist())

    k = 3
    similarity_threshold = 0.3  # temporarily lower
    topk_sim, topk_idx = torch.topk(similarities, k)
    topk_labels = [all_labels[i] for i in topk_idx]

    votes = {}
    for label, sim in zip(topk_labels, topk_sim):
        votes[label] = votes.get(label, 0.0) + sim.item()

    if not votes:
        return {"model": None, "confidence": 0.0}

    best_label = max(votes, key=votes.get)
    best_score = votes[best_label] / k

    if best_score < similarity_threshold:
        return {"model": None, "confidence": round(best_score, 4)}

    return {"model": best_label, "confidence": round(best_score, 4)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000, reload=True)
