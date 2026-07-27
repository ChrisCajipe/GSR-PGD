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
    MIXED_DIR,
    MODE,
    EVALUATION
)

from dataset.laion_loader import load_laion

from utils.preprocessing import preprocess_image
from utils.imagenet_labels import is_dog
from utils.visualization import save_image
from utils.dataset_builder import create_mixed_dataset, split_dataset
from utils.create_folders import create_directories

from attacks.pgd import targeted_pgd
from attacks.gsr_pgd import gsr_pgd

from models.resnet import (
    load_resnet,
    predict_image,
    evaluate_resnet
)

from models.lightshed_runner import (
    run_lightshed,
    summarize_lightshed
)

from models.trufor_runner import (
    run_trufor,
    summarize_trufor_results,
    save_trufor_heatmaps
)

from config import EVALUATION
from evaluation.metrics import calculate_binary_metrics
from evaluation.image_quality import evaluate_image_quality
from PIL import Image


# ==========================================================
# ATTACK
# ==========================================================

def generate_attacks(
    dataset,
    epsilon=EPSILON,
    alpha=ALPHA,
    max_iterations=MAX_ITERATIONS,
    sigma=SIGMA,
    lambda_reg=LAMBDA,
    attack=ATTACK
    ):

    # INITIALIZATION
    model = load_resnet()

    count = 0
    success = 0

    print("=" * 60)
    print(f"GENERATING {MODE.upper()} {EVALUATION.upper()} {attack.upper()} ATTACK")
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

        # ATTACK
        if attack == "pgd":
            adv_image, iterations = targeted_pgd(
                model=model,
                image=image,
                target_label=PERSIAN_CAT,
                epsilon=epsilon,
                alpha=alpha,
                max_iterations=max_iterations,
            )


        elif attack == "gsr":
            adv_image, iterations = gsr_pgd(
                model=model,
                image=image,
                target_label=PERSIAN_CAT,
                epsilon=epsilon,
                alpha=alpha,
                max_iterations=max_iterations,
                sigma=sigma,
                lambda_reg=lambda_reg,
            )

        # SAVE IMAGE
        save_image(image, ORIGINAL_DIR / f"{sample['image_id']}_original.png")
        save_image(adv_image, ADV_DIR / f"{sample['image_id']}_adv.png")

        count += 1

        adv_prediction_id, _ = predict_image(model, adv_image)

        if adv_prediction_id == PERSIAN_CAT:
            success += 1

        print(f"{sample['image_id']} | "f"Success {success}/{count}")

        if count >= MAX_IMAGES:
            break

    return count, success


# ==========================================================
# EVALUATION
# ==========================================================

def evaluate_defenses(attack=ATTACK):

    print("=" * 60)
    print(f"RUNNING {MODE.upper()} {EVALUATION.upper()} {attack.upper()} DEFENSE EVALUATION")
    print("=" * 60)

    if EVALUATION == "untampered":
        input_folder = ORIGINAL_DIR
        labels = [
            0
            for _ in input_folder.glob("*.png")
        ]


    elif EVALUATION == "tampered":
        input_folder = ADV_DIR
        labels = [
            1
            for _ in input_folder.glob("*.png")
        ]


    elif EVALUATION == "50-50":

        create_mixed_dataset(
            ORIGINAL_DIR,
            ADV_DIR,
            MIXED_DIR
        )

        input_folder = MIXED_DIR

        labels = []

        for image in sorted(input_folder.glob("*.png")):
            if "adv" in image.name:
                labels.append(1)
            else:
                labels.append(0)

    else:
        raise ValueError(
            "Invalid evaluation mode"
        )


    # ACTUAL START OF EVALUATION
    quality_results = None

    if EVALUATION in ["tampered", "50-50"]:
        resnet_results = evaluate_resnet()
        quality_results = evaluate_image_quality(ORIGINAL_DIR, ADV_DIR)
    else:
        resnet_results = None


    #running lightshed
    run_lightshed(input_folder,LIGHTSHED_DIR)

    #lightshed results
    lightshed_raw = summarize_lightshed(LIGHTSHED_DIR)
    lightshed_metrics = calculate_binary_metrics(
        predictions=lightshed_raw["predictions"],
        labels=labels
    )

    lightshed_results = {
        **lightshed_raw,
        **lightshed_metrics
    }

    #running trufor
    run_trufor(input_folder, TRUFOR_DIR)
    save_trufor_heatmaps(TRUFOR_DIR, input_folder)

    #trufor results
    trufor_raw = summarize_trufor_results(
        TRUFOR_DIR,
        threshold=0.5
    )

    trufor_metrics = calculate_binary_metrics(
        predictions=trufor_raw["predictions"],
        labels=labels
    )

    trufor_results = {
        **trufor_raw,
        **trufor_metrics
    }

    return resnet_results, quality_results, lightshed_results, trufor_results

def print_defense_summary(resnet_results, quality_results, lightshed_results, trufor_results):
    print("\n" + "=" * 60)
    print("DEFENSE EVALUATION SUMMARY")
    print(f"EVALUATING: {MODE.upper()} {EVALUATION.upper()} {ATTACK.upper()}")
    print("=" * 60)
    

    if resnet_results is not None:
        print("\nRESNET-50 PERFORMANCE")
        print("-" * 60)
        print(f"Images Evaluated    : {resnet_results['images']}")
        print(f"Successful Attacks  : {resnet_results['successful']}")
        print(f"Failed Attacks      : {resnet_results['failed']}")
        print(f"Attack Success Rate : {resnet_results['asr'] * 100:.2f}%")

    if quality_results is not None:
        print("\nIMAGE QUALITY")
        print("-" * 60)

        print(f"Average PSNR        : {quality_results['average_psnr']:.4f} dB")
        print(f"Average SSIM        : {quality_results['average_ssim']:.4f}")

    print("\nLIGHTSHED PERFORMANCE")
    print("-" * 60)
    print(f"Images Evaluated    : {lightshed_results['total']}")
    print(f"Flagged             : {lightshed_results['detected']}")
    print(f"Missed              : {lightshed_results['missed']}")

    if lightshed_results["accuracy"] is not None:
        print(
            f"Accuracy            : {lightshed_results['accuracy']*100:.2f}%"
        )

    if lightshed_results["tpr"] is not None:
        print(
            f"TPR                 : {lightshed_results['tpr']*100:.2f}%"
        )

    if lightshed_results["tnr"] is not None:
        print(
            f"TNR                 : {lightshed_results['tnr']*100:.2f}%"
        )

    print("\nTRUFOR PERFORMANCE")
    print("-" * 60)
    print(f"Images Evaluated    : {trufor_results['images']}")
    print(f"Average Score       : {trufor_results['average']:.4f}")
    print(f"Minimum Score       : {trufor_results['minimum']:.4f}")
    print(f"Maximum Score       : {trufor_results['maximum']:.4f}")

    if trufor_results["accuracy"] is not None:
        print(
            f"Accuracy            : {trufor_results['accuracy']*100:.2f}%"
        )

    if trufor_results["tpr"] is not None:
        print(
            f"TPR                 : {trufor_results['tpr']*100:.2f}%"
        )

    if trufor_results["tnr"] is not None:
        print(
            f"TNR                 : {trufor_results['tnr']*100:.2f}%"
        )
    print("=" * 60)
    print("\n")

# ==========================================================
# EXPERIMENT
# ==========================================================

