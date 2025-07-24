from torchvision import models, transforms
from PIL import Image
import torch
import torch.nn.functional as F
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained ResNet18 and strip classification head
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Identity()
model = model.to(device).eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def get_embedding(image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(tensor)
        embedding = F.normalize(embedding, p=2, dim=1)  # L2 normalize
    return embedding.squeeze().cpu()

def cosine_similarity(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
