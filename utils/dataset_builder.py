import shutil
import random
from pathlib import Path
import json

def build_evaluation_folder(image_dir, eval_dir, valid_ids, suffix):
    image_dir = Path(image_dir)
    eval_dir = Path(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    for file in eval_dir.glob("*"):
        file.unlink()

    images = filter_by_ids(sorted(image_dir.glob("*.png")), valid_ids, suffix)
    for image in images:
        shutil.copy(image, eval_dir / image.name)

def filter_by_ids(image_paths, valid_ids, suffix):
    """
    Keeps only images whose base image_id (with suffix stripped) is in valid_ids.
    suffix is e.g. "_original" or "_adv"
    """
    filtered = []
    for path in image_paths:
        stem = path.stem
        image_id = stem[: -len(suffix)] if stem.endswith(suffix) else stem
        if image_id in valid_ids:
            filtered.append(path)
    return filtered


def save_split_ids(tuning_ids, evaluation_ids, path):
    path = Path(path)
    with open(path, "w") as f:
        json.dump(
            {
                "tuning_ids": sorted(tuning_ids),
                "evaluation_ids": sorted(evaluation_ids),
            },
            f,
            indent=2,
        )


def load_evaluation_ids(path):
    path = Path(path)
    with open(path) as f:
        data = json.load(f)
    return set(data["evaluation_ids"])

def create_mixed_dataset(original_dir, adversarial_dir, mixed_dir, valid_ids=None):
    original_dir = Path(original_dir)
    adversarial_dir = Path(adversarial_dir)
    mixed_dir = Path(mixed_dir)

    mixed_dir.mkdir(parents=True, exist_ok=True)
    for file in mixed_dir.glob("*"):
        file.unlink()

    original_images = sorted(original_dir.glob("*.png"))
    adversarial_images = sorted(adversarial_dir.glob("*.png"))

    if valid_ids is not None:
        original_images = filter_by_ids(original_images, valid_ids, "_original")
        adversarial_images = filter_by_ids(adversarial_images, valid_ids, "_adv")

    count = min(len(original_images), len(adversarial_images))
    original_images = original_images[:count]
    adversarial_images = adversarial_images[:count]

    labels = []
    for image in original_images:
        shutil.copy(image, mixed_dir / image.name)
        labels.append(0)
    for image in adversarial_images:
        shutil.copy(image, mixed_dir / image.name)
        labels.append(1)

    return labels


def split_dataset(dataset, tuning_size=100, seed=42):

    indices = list(range(len(dataset)))

    random.seed(seed)
    random.shuffle(indices)

    tuning_indices = indices[:tuning_size]
    evaluation_indices = indices[tuning_size:]

    return tuning_indices, evaluation_indices