from datasets import load_dataset
from pathlib import Path
from PIL import Image, ImageOps
import random
import csv
import hashlib

# ============================================================
# SETTINGS
# ============================================================

dataset_path = "dataset/wikiart_poc"

DATASET_NAME = "huggan/wikiart"
SPLIT = "train"

OUTPUT_DIR = Path(dataset_path)

ARTWORKS_PER_ARTIST = 5
SELECTION_SEED = 2026

TARGET_ARTISTS = [
    "vincent-van-gogh",
    "rembrandt",
    "albrecht-durer",
]

TARGET_SIZE = (512, 512)


# ============================================================
# LOAD WIKIART
# ============================================================

print("Loading WikiArt...")

dataset = load_dataset(
    DATASET_NAME,
    split=SPLIT,
    streaming=True
)

# Artist, genre, and style are ClassLabel fields
artist_feature = dataset.features["artist"]
genre_feature = dataset.features["genre"]
style_feature = dataset.features["style"]


# ============================================================
# SHOW / RESOLVE ARTIST IDS
# ============================================================

artist_id_to_name = {
    i: artist_feature.int2str(i)
    for i in range(artist_feature.num_classes)
}

target_artist_ids = {}

for wanted_artist in TARGET_ARTISTS:

    matches = [
        (artist_id, artist_name)
        for artist_id, artist_name in artist_id_to_name.items()
        if artist_name.lower() == wanted_artist.lower()
    ]

    if len(matches) == 0:
        raise ValueError(
            f"Artist not found: {wanted_artist}"
        )

    artist_id, artist_name = matches[0]

    target_artist_ids[artist_id] = artist_name


print("\nSelected artists:")
print("-" * 60)

for artist_id, artist_name in sorted(target_artist_ids.items()):
    print(
        f"Artist ID {artist_id:3d} | {artist_name}"
    )


# ============================================================
# COLLECT CANDIDATE ARTWORKS
# ============================================================

candidates = {
    artist_id: []
    for artist_id in target_artist_ids
}

print("\nScanning WikiArt dataset...")

for dataset_index, sample in enumerate(dataset):

    artist_id = sample["artist"]

    if artist_id not in target_artist_ids:
        continue

    candidates[artist_id].append({
        "dataset_index": dataset_index,
        "image": sample["image"],
        "genre": sample["genre"],
        "style": sample["style"]
    })


print("\nCandidate counts:")

for artist_id, artworks in sorted(candidates.items()):
    print(
        f"{target_artist_ids[artist_id]}: "
        f"{len(artworks)} artworks"
    )


# ============================================================
# REPRODUCIBLY SELECT 5 PER ARTIST
# ============================================================

rng = random.Random(SELECTION_SEED)

selected_samples = []

for artist_id, artworks in sorted(candidates.items()):

    if len(artworks) < ARTWORKS_PER_ARTIST:
        raise ValueError(
            f"{target_artist_ids[artist_id]} only has "
            f"{len(artworks)} artworks."
        )

    # Deterministic ordering before random sampling
    artworks = sorted(
        artworks,
        key=lambda x: x["dataset_index"]
    )

    chosen = rng.sample(
        artworks,
        ARTWORKS_PER_ARTIST
    )

    chosen = sorted(
        chosen,
        key=lambda x: x["dataset_index"]
    )

    for artwork in chosen:

        selected_samples.append({
            "artist_id": artist_id,
            "artist_name": target_artist_ids[artist_id],
            "dataset_index": artwork["dataset_index"],
            "image": artwork["image"],
            "genre_id": artwork["genre"],
            "style_id": artwork["style"]
        })


# ============================================================
# PREPROCESS FUNCTION
# ============================================================

def resize_and_pad(image, target_size=(512, 512)):
    """
    Resize while preserving aspect ratio,
    then pad to exactly 512x512.
    """

    image = image.convert("RGB")

    image.thumbnail(
        target_size,
        Image.Resampling.LANCZOS
    )

    canvas = Image.new(
        "RGB",
        target_size,
        (0, 0, 0)
    )

    x = (target_size[0] - image.width) // 2
    y = (target_size[1] - image.height) // 2

    canvas.paste(
        image,
        (x, y)
    )

    return canvas


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

manifest_rows = []


# ============================================================
# SAVE SELECTED ARTWORKS
# ============================================================

print("\nSaving artworks...")

for sample in selected_samples:

    artist_id = sample["artist_id"]
    artist_name = sample["artist_name"]
    dataset_index = sample["dataset_index"]

    genre_id = sample["genre_id"]
    style_id = sample["style_id"]

    genre_name = genre_feature.int2str(genre_id)
    style_name = style_feature.int2str(style_id)

    artist_folder = (
        f"{artist_id:03d}_"
        + artist_name.replace("-", "_")
    )

    artist_dir = OUTPUT_DIR / artist_folder

    original_dir = artist_dir / "original"
    processed_dir = artist_dir / "processed_512"

    original_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    processed_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ----------------------------------------
    # Original image
    # ----------------------------------------

    original_image = sample["image"].convert("RGB")

    original_width = original_image.width
    original_height = original_image.height

    original_filename = (
        f"art_{dataset_index:06d}.png"
    )

    original_path = (
        original_dir
        / original_filename
    )

    original_image.save(original_path)

    # ----------------------------------------
    # Processed 512x512 image
    # ----------------------------------------

    processed_image = resize_and_pad(
        original_image,
        TARGET_SIZE
    )

    processed_filename = (
        f"art_{dataset_index:06d}_512.png"
    )

    processed_path = (
        processed_dir
        / processed_filename
    )

    processed_image.save(processed_path)

    # ----------------------------------------
    # SHA-256
    # ----------------------------------------

    with open(original_path, "rb") as file:
        original_sha256 = hashlib.sha256(
            file.read()
        ).hexdigest()

    with open(processed_path, "rb") as file:
        processed_sha256 = hashlib.sha256(
            file.read()
        ).hexdigest()

    # ----------------------------------------
    # Manifest
    # ----------------------------------------

    manifest_rows.append({
        "dataset": "WikiArt",
        "source": DATASET_NAME,
        "split": SPLIT,
        "dataset_index": dataset_index,

        "artist_id": artist_id,
        "artist_name": artist_name,

        "genre_id": genre_id,
        "genre_name": genre_name,

        "style_id": style_id,
        "style_name": style_name,

        "selection_seed": SELECTION_SEED,

        "original_width": original_width,
        "original_height": original_height,

        "original_file": str(original_path),
        "processed_file": str(processed_path),

        "original_sha256": original_sha256,
        "processed_sha256": processed_sha256
    })

    print(
        f"Saved: {artist_name} | "
        f"{style_name} | "
        f"{processed_path}"
    )


# ============================================================
# SAVE MANIFEST
# ============================================================

manifest_path = (
    OUTPUT_DIR
    / "wikiart_poc_manifest.csv"
)

with open(
    manifest_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    fieldnames = [
        "dataset",
        "source",
        "split",
        "dataset_index",

        "artist_id",
        "artist_name",

        "genre_id",
        "genre_name",

        "style_id",
        "style_name",

        "selection_seed",

        "original_width",
        "original_height",

        "original_file",
        "processed_file",

        "original_sha256",
        "processed_sha256"
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
    f"Total artworks selected: "
    f"{len(selected_samples)}"
)

print(
    f"Manifest saved to: "
    f"{manifest_path}"
)

print("\nArtworks per artist:")

for artist_id in sorted(target_artist_ids):

    count = sum(
        1
        for sample in selected_samples
        if sample["artist_id"] == artist_id
    )

    print(
        f"{target_artist_ids[artist_id]}: "
        f"{count}"
    )