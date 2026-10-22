from datasets import load_dataset
from pathlib import Path
import random
import csv
import hashlib

# ============================================================
# SETTINGS
# ============================================================

dataset_path = "dataset/imagenet_poc"

DATASET_NAME = "ILSVRC/imagenet-1k"
SPLIT = "validation"

OUTPUT_DIR = Path(dataset_path)

IMAGES_PER_CLASS = 5
SELECTION_SEED = 2026

# Selected non-dog / non-cat ImageNet classes
TARGET_CLASSES = [
    "monarch",
    "acoustic guitar",
    "church",
    "coffee mug",
    "mountain bike",
    "vase",
]


# ============================================================
# LOAD IMAGENET VALIDATION SET
# ============================================================

print("Loading ImageNet-1K validation set...")

dataset = load_dataset(
    DATASET_NAME,
    split=SPLIT,
    streaming=True
)

label_feature = dataset.features["label"]

# Map ImageNet label ID -> full class name
id_to_name = {
    i: label_feature.int2str(i)
    for i in range(label_feature.num_classes)
}


# ============================================================
# FIND EXACT IMAGENET CLASS IDS
# ============================================================

target_ids = {}

for wanted_name in TARGET_CLASSES:
    matches = []

    for class_id, full_name in id_to_name.items():

        # ImageNet names may contain multiple synonyms separated by commas
        synonyms = [
            name.strip().lower()
            for name in full_name.split(",")
        ]

        if wanted_name.lower() in synonyms:
            matches.append((class_id, full_name))

    if len(matches) == 0:
        raise ValueError(
            f"Could not find ImageNet class: {wanted_name}"
        )

    if len(matches) > 1:
        raise ValueError(
            f"Multiple matches found for '{wanted_name}': {matches}"
        )

    class_id, full_name = matches[0]

    target_ids[class_id] = {
        "short_name": wanted_name,
        "full_name": full_name
    }


print("\nSelected ImageNet classes:")
print("-" * 70)

for class_id, info in sorted(target_ids.items()):
    print(
        f"ID {class_id:3d} | {info['full_name']}"
    )


# ============================================================
# COLLECT VALIDATION IMAGES FOR TARGET CLASSES
# ============================================================

candidates = {
    class_id: []
    for class_id in target_ids
}

print("\nScanning ImageNet validation split...")

for validation_index, sample in enumerate(dataset):

    label = sample["label"]

    if label not in target_ids:
        continue

    candidates[label].append({
        "validation_index": validation_index,
        "image": sample["image"]
    })


print("\nCandidate counts:")

for class_id, items in sorted(candidates.items()):
    print(
        f"{class_id:3d} "
        f"{target_ids[class_id]['short_name']}: "
        f"{len(items)} images"
    )


# ============================================================
# SELECT 5 IMAGES PER CLASS
# ============================================================

rng = random.Random(SELECTION_SEED)

selected_samples = []

for class_id, items in sorted(candidates.items()):

    if len(items) < IMAGES_PER_CLASS:
        raise ValueError(
            f"Class {class_id} only contains "
            f"{len(items)} validation images."
        )

    # Sort first to make starting order deterministic
    items = sorted(
        items,
        key=lambda x: x["validation_index"]
    )

    # Fixed random sampling
    chosen = rng.sample(
        items,
        IMAGES_PER_CLASS
    )

    # Sort chosen samples for neat output
    chosen = sorted(
        chosen,
        key=lambda x: x["validation_index"]
    )

    for item in chosen:
        selected_samples.append({
            "class_id": class_id,
            "class_name": target_ids[class_id]["short_name"],
            "full_class_name": target_ids[class_id]["full_name"],
            "validation_index": item["validation_index"],
            "image": item["image"]
        })


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SAVE IMAGES
# ============================================================

manifest_rows = []

print("\nSaving selected images...")

for sample in selected_samples:

    class_id = sample["class_id"]
    class_name = sample["class_name"]
    validation_index = sample["validation_index"]

    # Example:
    # dataset/imagenet_poc/323_monarch/
    folder_name = (
        f"{class_id:03d}_"
        + class_name.replace(" ", "_")
    )

    class_dir = OUTPUT_DIR / folder_name

    class_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    filename = f"val_{validation_index:05d}.png"

    output_path = class_dir / filename

    # Convert to RGB for consistent image format
    image = sample["image"].convert("RGB")

    image.save(output_path)

    # Calculate SHA-256 checksum
    with open(output_path, "rb") as file:
        sha256 = hashlib.sha256(
            file.read()
        ).hexdigest()

    manifest_rows.append({
        "dataset": "ILSVRC ImageNet-1K",
        "split": SPLIT,
        "validation_index": validation_index,
        "class_id": class_id,
        "class_name": class_name,
        "full_class_name": sample["full_class_name"],
        "selection_seed": SELECTION_SEED,
        "filename": str(output_path),
        "sha256": sha256
    })

    print(f"Saved: {output_path}")


# ============================================================
# SAVE DATA MANIFEST
# ============================================================

manifest_path = (
    OUTPUT_DIR
    / "imagenet_poc_manifest.csv"
)

with open(
    manifest_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    fieldnames = [
        "dataset",
        "split",
        "validation_index",
        "class_id",
        "class_name",
        "full_class_name",
        "selection_seed",
        "filename",
        "sha256"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(manifest_rows)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(
    f"Dataset saved to: {OUTPUT_DIR}"
)

print(
    f"Total images selected: "
    f"{len(selected_samples)}"
)

print(
    f"Manifest saved to: "
    f"{manifest_path}"
)

print("\nImages per class:")

for class_id in sorted(target_ids):
    count = sum(
        1
        for sample in selected_samples
        if sample["class_id"] == class_id
    )

    print(
        f"{target_ids[class_id]['short_name']}: "
        f"{count}"
    )