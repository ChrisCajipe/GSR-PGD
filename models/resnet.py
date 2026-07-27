import torch
from torchvision.models import resnet50, ResNet50_Weights
from utils.preprocessing import preprocess_image, normalize_image
from pathlib import Path
from PIL import Image
from config import (
    ADV_DIR,
    PERSIAN_CAT
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

weights = ResNet50_Weights.IMAGENET1K_V2
categories = weights.meta["categories"]

def load_resnet():                # Loads the pretrained ImageNet-1k ResNet-50 model.

    model = resnet50(weights=weights)
    model.to(DEVICE)
    model.eval()
    return model

def predict_image(model, image):

    image = normalize_image(image)

    image = image.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(image)

    prediction_id = logits.argmax(dim=1).item()
    prediction_name = categories[prediction_id]

    return prediction_id, prediction_name

def evaluate_resnet():
    print("\nRunning ResNet-50 evaluation...")

    model = load_resnet()
    adv_images = list(ADV_DIR.glob("*.png"))
    print(f"Found {len(adv_images)} adversarial images")

    total = 0
    successful = 0

    for image_path in adv_images:

        image = preprocess_image(Image.open(image_path))
        prediction_id, _ = predict_image(model,image)
        total += 1

        if prediction_id == PERSIAN_CAT:
            successful += 1

    

    return {
        "images": total,
        "successful": successful,
        "failed": total - successful,
        "asr":
            successful / total
            if total > 0
            else 0
    }