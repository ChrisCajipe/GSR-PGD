from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent

LIGHTSHED_DIR = PROJECT_ROOT / "lightshade-artifact" / "lightshed"
LIGHTSHED_SCRIPT = LIGHTSHED_DIR / "standard_detector.py"


def run_lightshed(input_folder, output_folder, batch_size=8):
    """
    Run the official LightShed detector.

    Parameters
    ----------
    input_folder : str or Path
        Folder containing PGD/GSR-PGD images.

    output_folder : str or Path
        Folder where LightShed outputs will be saved.
    """

    input_folder = Path(input_folder).resolve()
    output_folder = Path(output_folder).resolve()

    command = [
        sys.executable,                     # use the same venv
        str(LIGHTSHED_SCRIPT),
        "--mode", "from_images",
        "--input_folders", str(input_folder),
        "--output_dir", str(output_folder),
        "--batch_size", str(batch_size)
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