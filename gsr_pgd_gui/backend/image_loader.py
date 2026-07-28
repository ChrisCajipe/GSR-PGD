"""
backend/image_loader.py

Utilities for loading and preparing user-uploaded images.
"""

from PIL import Image
from torchvision import transforms

IMAGE_SIZE = (512, 512)

preprocess = transforms.Compose([
    transforms.Resize((512, 512)),
])


def load_image(path: str) -> Image.Image:
    """
    Load an image, convert to RGB, and resize to 512x512.
    """
    image = Image.open(path).convert("RGB")
    image = preprocess(image)
    return image