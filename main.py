# ==========================================================
# IMPORTS
# ==========================================================

import pandas as pd

from config import (
    ATTACK,
    PERSIAN_CAT,
    MAX_IMAGES,
    EPSILON,
    ALPHA,
    MAX_ITERATIONS,
    SIGMA,
    LAMBDA,
    ORIGINAL_DIR,
    ADV_DIR,
    RESULTS_DIR,
    LIGHTSHED_DIR,
)

from dataset.laion_loader import load_laion
from models.resnet import load_resnet, predict_image

from utils.preprocessing import preprocess_image
from utils.imagenet_labels import is_dog
from utils.visualization import save_image

from attacks.pgd import targeted_pgd
from attacks.gsr_pgd import gsr_pgd

from models.lightshed_runner import run_lightshed


# ==========================================================
# INITIALIZATION
# ==========================================================

dataset = load_laion()
model = load_resnet()


# Create folders
ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)
ADV_DIR.mkdir(parents=True, exist_ok=True)

count = 0
success = 0


print("=" * 60)
print(f"RUNNING {ATTACK.upper()} ATTACK")
print("=" * 60)


# ==========================================================
# MAIN LOOP
# ==========================================================

for sample in dataset:

    # ---------------- PREPROCESS ----------------

    image = preprocess_image(sample["image"])


    # ---------------- IMAGE CLASSIFICATION ----------------

    prediction_id, prediction_name = predict_image(
        model,
        image
    )


    # ---------------- DOG FILTER ----------------

    if not is_dog(prediction_id):
        continue


    # ---------------- ATTACK ----------------

    if ATTACK == "pgd":

        adv_image, iterations = targeted_pgd(
            model=model,
            image=image,
            target_label=PERSIAN_CAT,
            epsilon=EPSILON,
            alpha=ALPHA,
            max_iterations=MAX_ITERATIONS,
        )


    elif ATTACK == "gsr":

        adv_image, iterations = gsr_pgd(
            model=model,
            image=image,
            target_label=PERSIAN_CAT,
            epsilon=EPSILON,
            alpha=ALPHA,
            max_iterations=MAX_ITERATIONS,
            sigma=SIGMA,
            lambda_reg=LAMBDA,
        )


    else:
        raise ValueError(
            "ATTACK must be 'pgd' or 'gsr'."
        )


    # ---------------- SAVE IMAGES ----------------

    save_image(
        image,
        ORIGINAL_DIR / f"{sample['image_id']}_original.png"
    )

    save_image(
        adv_image,
        ADV_DIR / f"{sample['image_id']}_adv.png"
    )

    # ---------------- EVALUATION ----------------

    adv_prediction_id, adv_prediction_name = predict_image(
        model,
        adv_image
    )


    count += 1

    if adv_prediction_id == PERSIAN_CAT:
        success += 1


    print("-" * 60)
    print(f"Image           : {sample['image_id']}")
    print(f"Original Label  : {prediction_name}")
    print("Target Label    : Persian Cat")
    print(f"Adversarial     : {adv_prediction_name}")
    print(f"Iterations Used : {iterations}")
    print(f"Attack Success  : {success}/{count}")


    if count >= MAX_IMAGES:
        break



# ==========================================================
# LIGHTSHED
# ==========================================================

run_lightshed(
    input_folder=ADV_DIR,
    output_folder=LIGHTSHED_DIR
)


# ==========================================================
# LIGHTSHED EVALUATION
# ==========================================================

csv_files = list(LIGHTSHED_DIR.glob("detection_analytics_*.csv"))

if csv_files:

    # Get latest LightShed output
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

    detected = df["is_poisoned"].sum()
    missed = total_images - detected


# ==========================================================
# FINAL SUMMARY
# ==========================================================

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(f"Attack Type         : {ATTACK.upper()}")

# Attack Results
print("\nATTACK PERFORMANCE")
print("-" * 60)

print(f"Images Tested       : {count}")
print(f"Successful Attacks  : {success}")

if count > 0:
    print(
        f"Attack Success Rate : "
        f"{100 * success / count:.2f}%"
    )
else:
    print("Attack Success Rate : 0%")


# LightShed Results
print("\nLIGHTSHED PERFORMANCE")
print("-" * 60)

if latest_csv:

    print(f"CSV Used            : {latest_csv.name}")
    print(f"Images Evaluated    : {total_images}")
    print(f"Flagged by LightShed: {detected}")
    print(f"Missed by LightShed : {missed}")

    print(
        f"Detection Rate      : "
        f"{100 * detected / total_images:.2f}%"
    )

    print(
        f"Evasion Rate        : "
        f"{100 * missed / total_images:.2f}%"
    )

else:
    print("No LightShed analytics CSV found.")


print("=" * 60)