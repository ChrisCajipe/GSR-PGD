import sys
from pathlib import Path

# allow importing project files
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))


import pandas as pd
from pipeline import generate_attacks, evaluate_defenses
from utils.create_folders import create_directories


# ==========================
# SEARCH SPACE
# ==========================

EPSILONS = [
    4 / 255,
    8 / 255,
    16 / 255
]

ALPHAS = [
    0.001,
    0.004,
    0.008,
    0.010,
    0.020
]

ITERATIONS = 10


# ==========================
# OUTPUT
# ==========================

RESULT_FILE = (
    PROJECT_ROOT
    / "tuning"
    / "results"
    / "pgd_tuning_results.csv"
)

results = []

# ==========================
# GRID SEARCH
# ==========================

for epsilon in EPSILONS:
    for alpha in ALPHAS:

        print("\n" + "="*70)
        print(
            f"TESTING "
            f"epsilon={epsilon:.5f}, "
            f"alpha={alpha:.5f}, "
            f"iterations={ITERATIONS}"
        )
        print("="*70)

        # clean folders
        create_directories()


        # generate PGD
        count, success = generate_attacks(
            epsilon=epsilon,
            alpha=alpha,
            max_iterations=ITERATIONS,
            attack="pgd"
        )

        # evaluate defenses
        (
            resnet_results,
            quality_results,
            lightshed_results,
            trufor_results
        ) = evaluate_defenses(attack="pgd")



        row = {
            "epsilon": epsilon,
            "alpha": alpha,
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