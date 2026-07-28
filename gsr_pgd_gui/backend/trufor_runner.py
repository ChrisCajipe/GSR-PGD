# backend/trufor_runner.py

from pathlib import Path
import subprocess
import shutil
import sys

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from backend.ui_config import (
    TRUFOR_DIR,
    TRUFOR_TEST,
    TRUFOR_CHECKPOINT,
)


def run_trufor(
    input_folder,
    output_folder,
):
    """
    Runs TruFor on every image inside input_folder.

    Returns the parsed results.
    """

    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    if output_folder.exists():
        shutil.rmtree(output_folder)

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        sys.executable,
        str(TRUFOR_TEST),
        "-in",
        str(input_folder),
        "-out",
        str(output_folder),
        "-exp",
        "trufor_ph3",
        "TEST.MODEL_FILE",
        str(TRUFOR_CHECKPOINT),
    ]

    print("\nRunning TruFor...")

    subprocess.run(
        command,
        cwd=TRUFOR_DIR,
        check=True,
    )

    print("TruFor complete.")

    return read_trufor_results(
        output_folder,
        input_folder,
    )


def read_trufor_results(
    result_folder,
    image_folder,
    threshold=0.5,
):
    """
    Reads TruFor outputs and saves
    the heatmap + overlay.

    Returns
    -------
    {
        "prediction": bool,
        "score": float,
        "heatmap": Path,
        "overlay": Path,
    }
    """

    result_folder = Path(result_folder)
    image_folder = Path(image_folder)

    npz_files = sorted(
        result_folder.rglob("*.npz")
    )

    if not npz_files:
        raise FileNotFoundError(
            "No TruFor .npz results found."
        )

    npz_file = npz_files[0]

    data = np.load(npz_file)

    score = float(data["score"])
    heatmap = data["map"]

    prediction = score >= threshold

    #
    # locate input image
    #

    image_files = (
        list(image_folder.glob("*.png"))
        + list(image_folder.glob("*.jpg"))
        + list(image_folder.glob("*.jpeg"))
    )

    if not image_files:
        raise FileNotFoundError(
            f"No images found inside {image_folder}"
        )

    image_path = image_files[0]
    image_name = image_path.stem
    image = np.array(
        Image.open(image_path)
        .convert("RGB")
    )

    #
    # output folders
    #

    maps_dir = result_folder / "maps"
    overlays_dir = result_folder / "overlays"

    maps_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    overlays_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    heatmap_file = maps_dir / f"{image_name}.png"
    overlay_file = overlays_dir / f"{image_name}.png"

    #
    # save heatmap
    #

    plt.figure(figsize=(6, 6))

    plt.imshow(
        heatmap,
        cmap="jet",
        vmin=0,
        vmax=1,
    )

    plt.axis("off")

    plt.savefig(
        heatmap_file,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
    )

    plt.close()

    #
    # save overlay
    #

    plt.figure(figsize=(6, 6))

    plt.imshow(image)

    plt.imshow(
        heatmap,
        cmap="jet",
        alpha=0.45,
        vmin=0,
        vmax=1,
    )

    plt.axis("off")

    plt.savefig(
        overlay_file,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
    )

    plt.close()

    return {
        "prediction": prediction,
        "score": score,
        "heatmap": heatmap_file,
        "overlay": overlay_file,
    }