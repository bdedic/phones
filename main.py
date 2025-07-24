from fastapi import FastAPI, UploadFile, File
from app.utils import get_embedding
import pickle
import torch
import io

app = FastAPI()

# Load your reference embeddings
with open("reference_embeddings.pkl", "rb") as f:
    reference_data = pickle.load(f)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    # Save temporary image
    temp_path = "temp.jpg"
    with open(temp_path, "wb") as f:
        f.write(contents)

    # Get embedding for uploaded image
    embedding = get_embedding(temp_path)

    # Find closest match (e.g., cosine similarity)
    best_match = None
    best_score = -1
    for label, ref_emb in reference_data.items():
        sim = torch.nn.functional.cosine_similarity(embedding, ref_emb, dim=0).item()
        if sim > best_score:
            best_score = sim
            best_match = label

    return {"model": best_match, "confidence": round(best_score, 4)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000, reload=True)
