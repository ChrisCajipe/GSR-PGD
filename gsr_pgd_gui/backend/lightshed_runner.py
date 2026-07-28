from pathlib import Path
import subprocess
import sys
import shutil
import pandas as pd

from backend.ui_config import (
    LIGHTSHED_CODE_DIR,
    LIGHTSHED_SCRIPT,
)


def run_lightshed(
    input_folder,
    output_folder,
    batch_size=8,
):
    """
    Run the official LightShed detector.

    Returns the output directory.
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
        str(LIGHTSHED_SCRIPT),
        "--mode",
        "from_images",
        "--input_folders",
        str(input_folder),
        "--output_dir",
        str(output_folder),
        "--batch_size",
        str(batch_size),
    ]

    subprocess.run(
        command,
        cwd=LIGHTSHED_CODE_DIR,
        check=True,
    )

    return read_lightshed_results(output_folder)


def read_lightshed_results(output_folder):
    """
    Read LightShed outputs.

    Returns
    -------
    {
        "detected": bool,
        "csv": Path,
        "prediction": bool,
        "entropy": float,
        "extracted_perturbation": Path | None,
    }
    """

    output_folder = Path(output_folder)

    csv_files = list(
        output_folder.glob("detection_analytics_*.csv")
    )

    if not csv_files:
        raise FileNotFoundError(
            "LightShed did not generate an analytics CSV."
        )

    csv_file = max(
        csv_files,
        key=lambda f: f.stat().st_mtime,
    )

    df = pd.read_csv(csv_file)

    row = df.iloc[0]

    prediction = str(
        row["is_poisoned"]
    ).lower().find("true") != -1

    entropy = float(row["entropy"])

    extracted = None

    candidates = list(
        output_folder.glob("poison_patterns/*.png")
    )
    
    extracted = None

    if candidates:
        extracted = candidates[0]

    return {
        "detected": prediction,
        "csv": csv_file,
        "prediction": prediction,
        "entropy": entropy,
        "extracted_perturbation": extracted,
    }


def evaluate_lightshed(
    input_folder,
):
    """
    Convenience wrapper.

    Runs LightShed and immediately returns the parsed results.
    """

    output = run_lightshed(input_folder)

    return read_lightshed_results(output)