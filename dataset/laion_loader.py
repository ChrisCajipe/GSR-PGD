from pathlib import Path
from datasets import Dataset
from PIL import Image, PngImagePlugin

PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024  # 100 MB

def load_laion(
    dataset_path="dataset/verified_dogs"
):

    image_paths = sorted(
        Path(dataset_path).glob("*.png")
    )

    data = {
        "image": [],
        "image_id": [],
        "path": []
    }

    for image_path in image_paths:

        image = Image.open(image_path).convert("RGB")

        data["image"].append(image)
        data["image_id"].append(
            image_path.stem
        )
        data["path"].append(
            str(image_path)
        )

    return Dataset.from_dict(data)