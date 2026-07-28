from PIL import Image
from torchvision import transforms

IMAGE_SIZE = (512, 512)

resize = transforms.Resize(IMAGE_SIZE)

to_tensor = transforms.ToTensor()

def load_image(path: str):
    # Original image
    pil_image = Image.open(path).convert("RGB")

    # Resize once
    pil_image = resize(pil_image)

    # Tensor version for the backend
    tensor_image = to_tensor(pil_image)

    return {
    "pil": pil_image,
    "tensor": tensor_image,
    }