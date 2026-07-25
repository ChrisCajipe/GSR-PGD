import os
import subprocess
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

from config import (
    TRUFOR_TEST,
    TRUFOR_CHECKPOINT
)


def run_trufor(input_folder, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    command = [
        sys.executable,   # <-- use current venv Python
        str(TRUFOR_TEST),
        "-in",
        str(input_folder),
        "-out",
        str(output_folder),
        "-exp",
        "trufor_ph3",
        "TEST.MODEL_FILE",
        str(TRUFOR_CHECKPOINT)
    ]

    print("\nRunning TruFor...")
    #print("Python:", sys.executable)
    #print("Command:")
    #print(" ".join(command))

    subprocess.run(
        command,
        check=True,
        cwd=TRUFOR_TEST.parent
    )

    print("TruFor completed.")



def summarize_trufor_results(result_folder, display=True):

    result_folder = Path(result_folder)

    npz_files = list(result_folder.rglob("*.npz"))

    if not npz_files:
        print("No TruFor results found.")
        return None


    scores = []

    for file in npz_files:

        data = np.load(file)

        scores.append(
            float(data["score"])
        )


    results = {
        "images": len(scores),
        "average": np.mean(scores),
        "minimum": np.min(scores),
        "maximum": np.max(scores)
    }


    return results

def save_trufor_heatmaps(
    result_folder,
    image_folder,
):

    result_folder = Path(result_folder)
    image_folder = Path(image_folder)

    maps_dir = result_folder / "maps"
    overlays_dir = result_folder / "overlays"

    maps_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(result_folder.glob("*.npz"))

    print("\nSaving TruFor heatmaps...")

    for npz_file in npz_files:

        image_name = npz_file.stem          # image.png
        image_path = image_folder / image_name

        if not image_path.exists():
            print(f"Missing image: {image_name}")
            continue

        data = np.load(npz_file)

        heatmap = data["map"]
        score = float(data["score"])

        image = np.array(Image.open(image_path))

        #
        # Save raw heatmap
        #

        plt.figure(figsize=(6,6))

        plt.imshow(
            heatmap,
            cmap="jet",
            vmin=0,
            vmax=1
        )

        plt.colorbar(label="Manipulation Probability")
        plt.title(f"Score = {score:.3f}")
        plt.axis("off")

        plt.savefig(
            maps_dir / image_name,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0
        )

        plt.close()

        #
        # Save overlay
        #

        plt.figure(figsize=(6,6))

        plt.imshow(image)

        plt.imshow(
            heatmap,
            cmap="jet",
            alpha=0.45,
            vmin=0,
            vmax=1
        )

        plt.title(f"Score = {score:.3f}")
        plt.axis("off")

        plt.savefig(
            overlays_dir / image_name,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0
        )

        plt.close()

    print(f"Saved {len(npz_files)} heatmaps.")