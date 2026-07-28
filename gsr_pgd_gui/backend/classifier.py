"""
backend/classifier.py

ResNet-50 ImageNet classifier.
"""

import torch
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
)

weights = ResNet50_Weights.IMAGENET1K_V2
model = resnet50(weights=weights)
model.eval()

preprocess = weights.transforms()

categories = weights.meta["categories"]


def predict_image(image):
    """
    Predict the ImageNet class of a PIL image.

    Returns:
        (class_name, confidence)
    """

    x = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        output = model(x)

    probabilities = torch.softmax(output, dim=1)

    confidence, prediction = probabilities.max(dim=1)

    return (
        categories[prediction.item()],
        confidence.item(),
    )