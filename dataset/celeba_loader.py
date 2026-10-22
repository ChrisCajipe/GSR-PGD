from datasets import load_dataset
from pathlib import Path
from PIL import Image
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
TARGET_SIZE = (512, 512)

TARGET_ARTISTS = [
    "vincent-van-gogh",
    "rembrandt",
    "albrecht-durer",
]


# ============================================================
# LOAD WIKIART
# ============================================================

print("Loading WikiArt...")

dataset = load_dataset(
    DATASET_NAME,
    split=SPLIT,
    streaming=True
)

artist_feature = dataset.features["artist"]
genre_feature = dataset.features["genre"]
style_feature = dataset.features["style"]


# ============================================================
# RESOLVE ARTIST IDS
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

    if len(matches) > 1:
        raise ValueError(
            f"Multiple matches found for: {wanted_artist}"
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
# STORAGE
# ============================================================

selected = {
    artist_id: []
    for artist_id in target_artist_ids
}


# ============================================================
# SCAN UNTIL EACH ARTIST HAS 5 ARTWORKS
# ============================================================

print("\nScanning WikiArt dataset...")

for dataset_index, sample in enumerate(dataset):

    artist_id = sample["artist"]

    # Ignore artists we are not using
    if artist_id not in target_artist_ids:
        continue

    # Ignore this artist if we already have 5
    if len(selected[artist_id]) >= ARTWORKS_PER_ARTIST:
        continue

    selected[artist_id].append({
        "dataset_index": dataset_index,
        "image": sample["image"],
        "genre": sample["genre"],
        "style": sample["style"]
    })

    artist_name = target_artist_ids[artist_id]

    print(
        f"Found {artist_name}: "
        f"{len(selected[artist_id])}/{ARTWORKS_PER_ARTIST}"
    )

    # Stop immediately once all artists have 5
    if all(
        len(selected[aid]) >= ARTWORKS_PER_ARTIST
        for aid in target_artist_ids
    ):
        print("\nAll required artworks found.")
        break


# ============================================================
# CHECK RESULTS
# ============================================================

for artist_id in target_artist_ids:

    if len(selected[artist_id]) < ARTWORKS_PER_ARTIST:
        raise ValueError(
            f"Only found {len(selected[artist_id])} artworks for "
            f"{target_artist_ids[artist_id]}"
        )


# ============================================================
# RESIZE + PAD FUNCTION
# ============================================================

def resize_and_pad(image, target_size=(512, 512)):

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
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

manifest_rows = []


# ============================================================
# SAVE ARTWORKS
# ============================================================

print("\nSaving artworks...")

for artist_id, artworks in sorted(selected.items()):

    artist_name = target_artist_ids[artist_id]

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

    for number, artwork in enumerate(
        artworks,
        start=1
    ):

        dataset_index = artwork["dataset_index"]

        genre_id = artwork["genre"]
        style_id = artwork["style"]

        genre_name = genre_feature.int2str(genre_id)
        style_name = style_feature.int2str(style_id)

        # ----------------------------------------
        # Original
        # ----------------------------------------

        original_image = artwork["image"].convert("RGB")

        original_width = original_image.width
        original_height = original_image.height

        original_filename = (
            f"art_{number:02d}_"
            f"idx_{dataset_index:06d}.png"
        )

        original_path = (
            original_dir
            / original_filename
        )

        original_image.save(original_path)

        # ----------------------------------------
        # Processed 512x512
        # ----------------------------------------

        processed_image = resize_and_pad(
            original_image,
            TARGET_SIZE
        )

        processed_filename = (
            f"art_{number:02d}_"
            f"idx_{dataset_index:06d}_512.png"
        )

        processed_path = (
            processed_dir
            / processed_filename
        )

        processed_image.save(processed_path)

        # ----------------------------------------
        # SHA-256 hashes
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

            "selection_method":
                "first five eligible artworks in dataset order",

            "original_width": original_width,
            "original_height": original_height,

            "original_file": str(original_path),
            "processed_file": str(processed_path),

            "original_sha256": original_sha256,
            "processed_sha256": processed_sha256
        })

        print(
            f"Saved: {artist_name} | "
            f"{number}/5 | "
            f"{style_name}"
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

        "selection_method",

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

print(f"Dataset saved to: {OUTPUT_DIR}")
print(f"Total artworks selected: {len(manifest_rows)}")
print(f"Manifest saved to: {manifest_path}")

print("\nArtworks per artist:")

for artist_id in sorted(target_artist_ids):

    print(
        f"{target_artist_ids[artist_id]}: "
        f"{len(selected[artist_id])}"
    )