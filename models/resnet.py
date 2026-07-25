import torch
from torchvision.models import resnet50, ResNet50_Weights
from utils.preprocessing import normalize_image

weights = ResNet50_Weights.IMAGENET1K_V2
categories = weights.meta["categories"]

def load_resnet():                # Loads the pretrained ImageNet-1k ResNet-50 model.

    model = resnet50(weights=weights)
    model.eval()
    return model

def predict_image(model, image):    # Predicts the ImageNet class of one preprocessed image.
    
    image = normalize_image(image)

    image = image.unsqueeze(0)

    with torch.no_grad():
        logits = model(image)

    prediction_id = logits.argmax(dim=1).item()
    prediction_name = categories[prediction_id]

    return prediction_id, prediction_name