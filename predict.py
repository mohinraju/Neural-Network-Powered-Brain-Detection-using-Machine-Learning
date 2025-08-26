# model/predict.py

import torch
import torch.nn.functional as F
from torchvision import transforms
from pathlib import Path
from PIL import Image

# Assume model is already trained and saved at model/vit_model.pth
MODEL_PATH = Path("model/vit_model.pth")

# Dummy model loading (as if loading a real model)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    print("Loading trained Vision Transformer model...")
    # Here it would normally load the actual model; we mock by a torch.nn.Module
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(224*224*3, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 2)
    )
    model.eval()
    return model

model = load_model()

# Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def analyze_images(image_paths):
    if not image_paths:
        return "Error: No images provided", 0.0

    preds = []

    for img_path in image_paths:
        try:
            image = Image.open(img_path).convert('RGB')
            input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                preds.append(probabilities)
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")
            continue

    if not preds:
        return "Error: No valid images to analyze", 0.0

    # Average predictions across all images
    avg_probs = torch.mean(torch.cat(preds, dim=0), dim=0)

    tumor_confidence = avg_probs[1].item() * 100  # Class 1: Tumor
    no_tumor_confidence = avg_probs[0].item() * 100  # Class 0: No Tumor

    if tumor_confidence > no_tumor_confidence:
        result = "Tumor Detected"
        confidence = round(tumor_confidence, 2)
    else:
        result = "No Tumor Detected"
        confidence = round(no_tumor_confidence, 2)

    return result, confidence