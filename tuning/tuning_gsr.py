import sys
from pathlib import Path

# allow importing project files
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
from pipeline import generate_attacks, evaluate_defenses
from utils.create_folders import create_directories
from utils.dataset_builder import split_dataset
from dataset.laion_loader import load_laion
from config import EPSILON, ALPHA

dataset = load_laion()
tuning_indices, evaluation_indices = split_dataset(dataset,tuning_size=100)
tuning_dataset = dataset.select(tuning_indices)

# ==========================
# SEARCH SPACE
# ==========================

SIGMAS = [
    0.01,
    0.03,
    0.05,
    0.07,
    0.10
]

LAMBDAS = [
    0.01,
    0.05,
    0.10,
    0.20,
    0.50,
    0.75
]

ITERATIONS = 10

# ==========================
# OUTPUT
# ==========================

RESULT_FILE = (
    PROJECT_ROOT
    / "tuning"
    / "results"
    / "gsr_tuning_results.csv"
)

results = []

# ==========================
# GRID SEARCH
# ==========================

for sigma in SIGMAS:
    for lambda_reg in LAMBDAS:

        print("\n" + "="*70)
        print(
            f"TESTING "
            f"sigma={sigma:.5f}, "
            f"lambda={lambda_reg:.5f}, "
            f"iterations={ITERATIONS}"
        )
        print("="*70)

        # clean folders
        create_directories()

        # generate PGD
        count, success = generate_attacks(
            dataset=tuning_dataset,
            epsilon=EPSILON,
            alpha=ALPHA,
            max_iterations=ITERATIONS,
            attack="gsr",
            sigma=sigma,
            lambda_reg=lambda_reg
        )

        # evaluate defenses
        (
            resnet_results,
            quality_results,
            lightshed_results,
            trufor_results
        ) = evaluate_defenses(attack="pgd")



        row = {
            "sigma": sigma,
            "lambda": lambda_reg,
            "iterations": ITERATIONS,

            # attack
            "images": count,

            "ASR":
                resnet_results["asr"]
                if resnet_results
                else None,


            # quality
            "PSNR":
                quality_results["average_psnr"]
                if quality_results
                else None,

            "SSIM":
                quality_results["average_ssim"]
                if quality_results
                else None,


            # lightshed
            "LightShed_TPR":
                lightshed_results["tpr"],
            "LightShed_TNR":
                lightshed_results["tnr"],


            # trufor
            "TruFor_TPR":
                trufor_results["tpr"],
            "TruFor_TNR":
                trufor_results["tnr"],

        }

        results.append(row)
        pd.DataFrame(results).to_csv(RESULT_FILE,index=False)

        print("\nCURRENT RESULT")
        print(row)


print("\nFinished tuning.")
print(f"Saved results to {RESULT_FILE}")