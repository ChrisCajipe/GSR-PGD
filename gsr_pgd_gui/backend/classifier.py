"""
backend/classifier.py

ResNet-50 ImageNet classifier.
Supports:
- PIL images (uploaded images)
- Tensor images (PGD/GSR-PGD outputs)
"""

import torch
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
)

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from utils.preprocessing import normalize_image


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


weights = ResNet50_Weights.IMAGENET1K_V2

model = resnet50(
    weights=weights
).to(DEVICE)

model.eval()


categories = weights.meta["categories"]


# ---------------------------------------------------------
# PIL IMAGE PREDICTION
# Used for uploaded images
# ---------------------------------------------------------
def predict_image(image):
    """
    Predict ImageNet class from PIL Image.
    """

    preprocess = weights.transforms()

    x = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        output = model(x)


    probabilities = torch.softmax(
        output,
        dim=1
    )

    confidence, prediction = probabilities.max(
        dim=1
    )


    return (
        categories[prediction.item()],
        confidence.item(),
    )



# ---------------------------------------------------------
# TENSOR PREDICTION
# Used for PGD/GSR-PGD outputs
# ---------------------------------------------------------
def predict_tensor(image_tensor):
    """
    Predict ImageNet class from tensor.

    Expected input:
        Tensor [C,H,W]
        values in range [0,1]
    """


    image_tensor = image_tensor.detach().to(DEVICE)


    # Add batch dimension
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)


    # Normalize exactly like attack generation
    image_tensor = normalize_image(
        image_tensor
    )


    with torch.no_grad():

        output = model(
            image_tensor
        )


    probabilities = torch.softmax(
        output,
        dim=1
    )

    confidence, prediction = probabilities.max(
        dim=1
    )


    return (
        categories[prediction.item()],
        confidence.item(),
    )