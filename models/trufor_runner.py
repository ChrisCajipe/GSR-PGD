import os
import subprocess
import sys
import numpy as np
import matplotlib.pyplot as plt
import shutil

from PIL import Image
from pathlib import Path

from config import (
    TRUFOR_TEST,
    TRUFOR_CHECKPOINT
)


def run_trufor(input_folder, output_folder):

    output_folder = Path(output_folder)

    # clear old results
    if output_folder.exists():
        shutil.rmtree(output_folder)

    output_folder.mkdir(
        parents=True,
        exist_ok=True
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
        str(TRUFOR_CHECKPOINT)
    ]

    print("\n𝗥𝘂𝗻𝗻𝗶𝗻𝗴 𝗧𝗿𝘂𝗙𝗼𝗿...")

    subprocess.run(
        command,
        check=True,
        cwd=TRUFOR_TEST.parent
    )

    print("TruFor completed.")



def summarize_trufor_results(
    result_folder,
    threshold=0.5
):

    result_folder = Path(result_folder)

    npz_files = list(
        result_folder.rglob("*.npz")
    )

    if not npz_files:
        print("No TruFor results found.")
        return None


    scores = []
    predictions = []


    for file in npz_files:

        data = np.load(file)

        score = float(data["score"])

        scores.append(score)

        predictions.append(
            1 if score >= threshold else 0
        )


    return {
        "images": len(scores),

        # raw scores
        "scores": scores,

        # binary decisions
        "predictions": predictions,

        "average": np.mean(scores),
        "minimum": np.min(scores),
        "maximum": np.max(scores),
        "threshold": threshold
    }



def save_trufor_heatmaps(
    result_folder,
    image_folder,
):

    result_folder = Path(result_folder)
    image_folder = Path(image_folder)

    maps_dir = result_folder / "maps"
    overlays_dir = result_folder / "overlays"

    maps_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    overlays_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    npz_files = sorted(
        result_folder.rglob("*.npz")
    )

    print("\nSaving TruFor heatmaps...")


    saved = 0

    for npz_file in npz_files:

        # Remove only .npz
        image_name = npz_file.name.replace(
            ".npz",
            ""
        )

        image_path = image_folder / image_name


        if not image_path.exists():
            print(
                f"Missing image: {image_path.name}"
            )
            continue


        data = np.load(npz_file)

        heatmap = data["map"]
        score = float(data["score"])


        image = np.array(
            Image.open(image_path)
            .convert("RGB")
        )


        #
        # RAW HEATMAP
        #

        plt.figure(figsize=(6,6))

        plt.imshow(
            heatmap,
            cmap="jet",
            vmin=0,
            vmax=1
        )

        plt.colorbar(
            label="Manipulation Probability"
        )

        plt.title(
            f"Score = {score:.3f}"
        )

        plt.axis("off")

        plt.savefig(
            maps_dir / image_name,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0
        )

        plt.close()



        #
        # OVERLAY
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

        plt.title(
            f"Score = {score:.3f}"
        )

        plt.axis("off")

        plt.savefig(
            overlays_dir / image_name,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0
        )

        plt.close()


        saved += 1


    print(
        f"Saved {saved}/{len(npz_files)} TruFor heatmaps."
    )