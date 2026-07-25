from torchvision import transforms
import torch

preprocess = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

normalize = transforms.Compose([    # Derived from PyTorch documentation
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess_image(image):        # Resize & tensor
    return preprocess(image)        # Output range: [0,1]

def normalize_image(image):         # Normalize image using ImageNet stats
    return normalize(image)