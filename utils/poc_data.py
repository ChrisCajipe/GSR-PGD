import random
import json
from pathlib import Path


# Change this to your original 1,100-image folder
DATASET_DIR = Path(r"C:\Users\chris\School\Thesis\GSR-PGD\dataset\verified_dogs")

# IMPORTANT:
# This ordering must be the same ordering used originally.
dataset = sorted(DATASET_DIR.glob("*.png"))

print(f"Found {len(dataset)} images")

if len(dataset) != 1100:
    raise ValueError(
        f"Expected 1100 images, but found {len(dataset)}."
    )


# Reproduce original seed-42 split
indices = list(range(len(dataset)))

rng = random.Random(42)
rng.shuffle(indices)

tuning_indices = indices[:100]
evaluation_indices = indices[100:]


# Convert indices to actual filenames / IDs
tuning_images = [
    dataset[i]
    for i in tuning_indices
]

evaluation_images = [
    dataset[i]
    for i in evaluation_indices
]


# Your 16 PoC images
poc_images = tuning_images[:16]

print("\nPoC images:")
for image in poc_images:
    print(image.name)


# Save the recovered split again so you don't lose it
split_data = {
    "tuning_ids": [
        image.stem
        for image in tuning_images
    ],
    "evaluation_ids": [
        image.stem
        for image in evaluation_images
    ],
}

with open("dataset_split.json", "w") as f:
    json.dump(split_data, f, indent=2)

print("\nRecovered split saved to dataset_split.json")