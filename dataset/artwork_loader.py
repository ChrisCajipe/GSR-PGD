from pathlib import Path
from datasets import Dataset, Image as HFImage


def load_artworks(
    dataset_path="dataset/wikiart_poc"
):

    root = Path(dataset_path)

    if not root.exists():
        raise FileNotFoundError(
            f"Artwork dataset not found: {dataset_path}"
        )

    # Only load the standardized 512x512 artworks
    image_paths = sorted(
        path
        for path in root.rglob("*.png")
        if path.parent.name == "processed_512"
    )

    if not image_paths:
        raise ValueError(
            f"No processed artwork images found in {dataset_path}"
        )

    images = []
    image_ids = []
    artists = []

    for image_path in image_paths:

        # Example:
        # 022_vincent_van_gogh
        artist = image_path.parent.parent.name

        # Globally unique image ID
        image_id = (
            f"{artist}_{image_path.stem}"
        )

        images.append(
            str(image_path)
        )

        image_ids.append(
            image_id
        )

        artists.append(
            artist
        )

    dataset = Dataset.from_dict({
        "image": images,
        "image_id": image_ids,
        "artist": artists,
    })

    dataset = dataset.cast_column(
        "image",
        HFImage()
    )

    print(
        f"Loaded {len(dataset)} WikiArt images."
    )

    print("\nDataset composition:")

    for artist in sorted(set(artists)):

        count = artists.count(artist)

        print(
            f"  {artist}: {count}"
        )

    return dataset