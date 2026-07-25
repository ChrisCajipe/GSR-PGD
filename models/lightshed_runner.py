from pathlib import Path
import subprocess
import sys
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

LIGHTSHED_DIR = PROJECT_ROOT / "lightshade-artifact" / "lightshed"
LIGHTSHED_SCRIPT = LIGHTSHED_DIR / "standard_detector.py"


def run_lightshed(input_folder, output_folder, batch_size=8):
    """
    Run the official LightShed detector.
    """

    input_folder = Path(input_folder).resolve()
    output_folder = Path(output_folder).resolve()

    output_folder.mkdir(
        parents=True,
        exist_ok=True
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
        str(batch_size)
    ]

    print("\nRunning LightShed...")
    print("Input :", input_folder)
    print("Output:", output_folder)

    subprocess.run(
        command,
        cwd=LIGHTSHED_DIR,
        check=True
    )

    print("LightShed evaluation complete.")



def summarize_lightshed(output_folder, display=True):
    """
    Read LightShed CSV output and print detection statistics.
    """

    output_folder = Path(output_folder)

    csv_files = list(
        output_folder.glob(
            "detection_analytics_*.csv"
        )
    )

    if not csv_files:
        print("\nNo LightShed analytics CSV found.")
        return None


    latest_csv = max(
        csv_files,
        key=lambda x: x.stat().st_mtime
    )


    df = pd.read_csv(latest_csv)


    # Convert tensor(True)/tensor(False) strings
    df["is_poisoned"] = (
        df["is_poisoned"]
        .astype(str)
        .str.contains("True")
    )


    total_images = len(df)

    detected = int(
        df["is_poisoned"].sum()
    )

    missed = total_images - detected


    return {
        "csv": latest_csv,
        "total": total_images,
        "detected": detected,
        "missed": missed,
        "detection_rate": (
            detected / total_images
            if total_images > 0
            else 0
        ),
        "evasion_rate": (
            missed / total_images
            if total_images > 0
            else 0
        )
    }