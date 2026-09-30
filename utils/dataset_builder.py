import shutil
import random
from pathlib import Path
import json


# ==========================================================
# FILTERING / FOLDER BUILDING
# ==========================================================

def filter_by_ids(image_paths, valid_ids, suffix):
    """
    Keep only images whose base image_id is in valid_ids.

    suffix examples:
        "_original"
        "_adv"
    """

    valid_ids = set(valid_ids)

    filtered = []

    for path in image_paths:
        stem = path.stem

        image_id = (
            stem[:-len(suffix)]
            if stem.endswith(suffix)
            else stem
        )

        if image_id in valid_ids:
            filtered.append(path)

    return filtered


def build_evaluation_folder(
    image_dir,
    eval_dir,
    valid_ids,
    suffix,
):
    image_dir = Path(image_dir)
    eval_dir = Path(eval_dir)

    eval_dir.mkdir(parents=True, exist_ok=True)

    # Clear previous evaluation folder
    for file in eval_dir.glob("*"):
        if file.is_file():
            file.unlink()
        elif file.is_dir():
            shutil.rmtree(file)

    images = filter_by_ids(
        sorted(image_dir.glob("*.png")),
        valid_ids,
        suffix,
    )

    for image in images:
        shutil.copy(
            image,
            eval_dir / image.name
        )


# ==========================================================
# DATASET SPLITTING
# ==========================================================

def split_dataset(
    dataset,
    tuning_size=100,
    seed=42,
):
    """
    Create ONE deterministic randomized ordering.

    The first tuning_size samples are used for tuning.
    Everything after that is used for evaluation.

    IMPORTANT:
    Returned order is preserved so evaluation subsets
    can later be selected as:

        evaluation_ids[:10]
        evaluation_ids[:50]
        evaluation_ids[:100]
        ...

    This guarantees nested subsets.
    """

    indices = list(range(len(dataset)))

    # Local RNG so we don't modify global random state
    rng = random.Random(seed)
    rng.shuffle(indices)

    tuning_indices = indices[:tuning_size]
    evaluation_indices = indices[tuning_size:]

    return tuning_indices, evaluation_indices


# ==========================================================
# SAVE / LOAD SPLIT
# ==========================================================

def save_split_ids(
    tuning_ids,
    evaluation_ids,
    path,
):
    """
    Save IDs while PRESERVING their randomized order.

    Do NOT sort evaluation_ids here, because their order
    determines the fixed subsets used for scalability tests.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(path, "w") as f:
        json.dump(
            {
                "tuning_ids": list(tuning_ids),
                "evaluation_ids": list(evaluation_ids),
            },
            f,
            indent=2,
        )


def load_evaluation_ids(path):
    """
    Load evaluation IDs as an ordered list.

    Do NOT convert to a set because that destroys ordering.
    """

    path = Path(path)

    with open(path) as f:
        data = json.load(f)

    return data["evaluation_ids"]


def select_evaluation_ids(
    evaluation_ids,
    amount,
):
    """
    Select the first N samples from the fixed evaluation order.

    Example:
        10   -> first 10
        50   -> same 10 + 40 more
        100  -> same 50 + 50 more
    """

    if amount > len(evaluation_ids):
        raise ValueError(
            f"Requested {amount} evaluation images, "
            f"but only {len(evaluation_ids)} are available."
        )

    return evaluation_ids[:amount]


# ==========================================================
# 50/50 MIXED DATASET
# ==========================================================

def create_mixed_dataset(
    original_dir,
    adversarial_dir,
    mixed_dir,
    valid_ids=None,
    seed=42,
):
    original_dir = Path(original_dir)
    adversarial_dir = Path(adversarial_dir)
    mixed_dir = Path(mixed_dir)

    mixed_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Clear previous mixed dataset
    for file in mixed_dir.glob("*"):
        if file.is_file():
            file.unlink()
        elif file.is_dir():
            shutil.rmtree(file)

    if valid_ids is not None:

        # Copy so original ordered list is not modified
        ids = list(valid_ids)

        rng = random.Random(seed)
        rng.shuffle(ids)

        half = len(ids) // 2

        original_ids = set(ids[:half])
        adversarial_ids = set(ids[half:])

        original_images = filter_by_ids(
            sorted(original_dir.glob("*.png")),
            original_ids,
            "_original",
        )

        adversarial_images = filter_by_ids(
            sorted(adversarial_dir.glob("*.png")),
            adversarial_ids,
            "_adv",
        )

    else:
        original_images = sorted(
            original_dir.glob("*.png")
        )

        adversarial_images = sorted(
            adversarial_dir.glob("*.png")
        )

    labels = []

    for image in original_images:
        shutil.copy(
            image,
            mixed_dir / image.name
        )

        labels.append(0)

    for image in adversarial_images:
        shutil.copy(
            image,
            mixed_dir / image.name
        )

        labels.append(1)

    return labels