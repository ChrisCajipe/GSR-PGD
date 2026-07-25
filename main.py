# ==========================================================
# IMPORTS
# ==========================================================

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
    LIGHTSHED_DIR,
    TRUFOR_DIR,
    MODE
)

from dataset.laion_loader import load_laion
from models.resnet import load_resnet, predict_image

from utils.preprocessing import preprocess_image
from utils.imagenet_labels import is_dog
from utils.visualization import save_image

from attacks.pgd import targeted_pgd
from attacks.gsr_pgd import gsr_pgd

from models.lightshed_runner import (
    run_lightshed,
    summarize_lightshed
)

from models.trufor_runner import (
    run_trufor,
    summarize_trufor_results,
    save_trufor_heatmaps
)

from PIL import Image


# ==========================================================
# ATTACK
# ==========================================================

def generate_attacks():

    # INITIALIZATION
    dataset = load_laion()
    model = load_resnet()

    count = 0
    success = 0

    print("=" * 60)
    print(f"GENERATING {ATTACK.upper()} ATTACK")
    print("=" * 60)


    for sample in dataset:

        # PREPROCESSING
        image = preprocess_image(
            sample["image"]
        )

        # IMAGE CLASSIFICATION
        prediction_id, prediction_name = predict_image(
            model,
            image
        )

        # DOG FILTER
        if not is_dog(prediction_id):
            continue

        # ATTACK
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

        # SAVE IMAGE
        save_image(
            image,
            ORIGINAL_DIR /
            f"{sample['image_id']}_original.png"
        )


        save_image(
            adv_image,
            ADV_DIR /
            f"{sample['image_id']}_adv.png"
        )

        count += 1

        adv_prediction_id, _ = predict_image(
            model,
            adv_image
        )


        if adv_prediction_id == PERSIAN_CAT:
            success += 1


        print(
            f"{sample['image_id']} | "
            f"Success {success}/{count}"
        )


        if count >= MAX_IMAGES:
            break


    return count, success


# ==========================================================
# EVALUATION
# ==========================================================

# RESNET EVALUATION
def evaluate_resnet():

    print("\nRunning ResNet-50 evaluation...")

    model = load_resnet()

    adv_images = list(
        ADV_DIR.glob("*.png")
    )


    total = 0
    successful = 0

    predictions = []


    for image_path in adv_images:

        image = preprocess_image(
            Image.open(image_path)
        )

        prediction_id, prediction_name = predict_image(
            model,
            image
        )


        total += 1


        if prediction_id == PERSIAN_CAT:
            successful += 1


        predictions.append(
            prediction_name
        )


    return {
        "images": total,
        "successful": successful,
        "failed": total - successful,
        "asr": (
            successful / total
            if total > 0
            else 0
        )
    }

def evaluate_defenses():

    print("=" * 60)
    print("RUNNING DEFENSE EVALUATION")
    print(f"\nEVALUATING: {ATTACK.upper()}")
    print("=" * 60)

    # ==========================
    # RESNET-50 EVALUATION
    # ==========================

    resnet_results = evaluate_resnet()

    # ==========================
    # LIGHTSHED
    # ==========================

    run_lightshed(
        input_folder=ADV_DIR,
        output_folder=LIGHTSHED_DIR
    )

    lightshed_results = summarize_lightshed(
        LIGHTSHED_DIR,
        display=False
    )


    # ==========================
    # TRUFOR
    # ==========================

    run_trufor(
        input_folder=ADV_DIR,
        output_folder=TRUFOR_DIR
    )

    save_trufor_heatmaps(
        result_folder=TRUFOR_DIR,
        image_folder=ADV_DIR
    )

    trufor_results = summarize_trufor_results(
        TRUFOR_DIR,
        display=False
    )


    return resnet_results, lightshed_results, trufor_results

def print_defense_summary(resnet_results, lightshed_results, trufor_results):
    print("\n" + "=" * 60)
    print("DEFENSE EVALUATION SUMMARY")
    print("=" * 60)
    print(f"\nEVALUATING: {ATTACK.upper()}")

    print("\nRESNET-50 PERFORMANCE")
    print("-" * 60)
    print(f"Images Evaluated    : {resnet_results['images']}")
    print(f"Successful Attacks  : {resnet_results['successful']}")
    print(f"Failed Attacks      : {resnet_results['failed']}")
    print(f"Attack Success Rate : {resnet_results['asr'] * 100:.2f}%")

    print("\nLIGHTSHED PERFORMANCE")
    print("-" * 60)
    print(f"Images Evaluated    : {lightshed_results['total']}")
    print(f"Flagged             : {lightshed_results['detected']}")
    print(f"Missed              : {lightshed_results['missed']}")
    print(f"Detection Rate      : {lightshed_results['detection_rate'] * 100:.2f}%")
    print(f"Evasion Rate        : {lightshed_results['evasion_rate'] * 100:.2f}%")

    print("\nTRUFOR PERFORMANCE")
    print("-" * 60)
    print(f"Images Evaluated    : {trufor_results['images']}")
    print(f"Average Score       : {trufor_results['average']:.4f}")
    print(f"Minimum Score       : {trufor_results['minimum']:.4f}")
    print(f"Maximum Score       : {trufor_results['maximum']:.4f}")
    print("=" * 60)
    print("\n")

# ==========================================================
# EXPERIMENT
# ==========================================================

if MODE == "attack":
    count, success = generate_attacks()

elif MODE == "evaluate":
    resnet_results, lightshed_results, trufor_results = evaluate_defenses()
    print_defense_summary(resnet_results, lightshed_results, trufor_results)


elif MODE == "full":
    count, success = generate_attacks()
    lightshed_results, trufor_results = evaluate_defenses()
    print_defense_summary(lightshed_results, trufor_results)


else:
    raise ValueError(
        "MODE must be 'attack', 'evaluate', or 'full'"
    )