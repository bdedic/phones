from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from PIL import Image
import torch
import torch.nn.functional as F

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load EfficientNet-B4 with pre-trained weights
weights = EfficientNet_B4_Weights.IMAGENET1K_V1
model = efficientnet_b4(weights=weights)

# Remove classification head
model.classifier = torch.nn.Identity()
model = model.to(device).eval()

# Preprocessing pipeline recommended for EfficientNet-B4
preprocess = weights.transforms()

def get_embedding(image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(tensor)
        embedding = F.normalize(embedding, p=2, dim=1)  # L2 normalize
    return embedding.squeeze().cpu()

def cosine_similarity(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
