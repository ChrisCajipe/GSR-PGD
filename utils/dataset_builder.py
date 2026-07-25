import shutil
from pathlib import Path


def create_mixed_dataset(
    original_dir,
    adversarial_dir,
    mixed_dir
):
    """
    Creates a 50-50 dataset containing:
    - 50% untampered images
    - 50% tampered images

    Labels:
        0 = untampered
        1 = tampered
    """

    original_dir = Path(original_dir)
    adversarial_dir = Path(adversarial_dir)
    mixed_dir = Path(mixed_dir)


    mixed_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # remove previous mixed dataset
    for file in mixed_dir.glob("*"):
        file.unlink()


    original_images = sorted(
        original_dir.glob("*.png")
    )

    adversarial_images = sorted(
        adversarial_dir.glob("*.png")
    )


    count = min(
        len(original_images),
        len(adversarial_images)
    )


    original_images = original_images[:count]
    adversarial_images = adversarial_images[:count]


    labels = []


    # Copy untampered
    for image in original_images:

        shutil.copy(
            image,
            mixed_dir / image.name
        )

        labels.append(0)


    # Copy tampered
    for image in adversarial_images:

        shutil.copy(
            image,
            mixed_dir / image.name
        )

        labels.append(1)


    print("\nMixed dataset created")
    print(f"Untampered images : {count}")
    print(f"Tampered images   : {count}")
    print(f"Total images      : {len(labels)}")


    return labels