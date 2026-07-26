from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import *
from dataset.laion_loader import load_laion
from utils.preprocessing import preprocess_image
from utils.imagenet_labels import is_dog
from attacks.pgd import targeted_pgd
from models.resnet import load_resnet, predict_image

MAX_ITERATIONS = 100

dataset = load_laion()
model = load_resnet()

iterations_used = []
success = 0
count = 0

for sample in dataset:

    image = preprocess_image(sample["image"])

    pred_id, _ = predict_image(model, image)

    if not is_dog(pred_id):
        continue

    adv_image, used_iterations = targeted_pgd(
        model=model,
        image=image,
        target_label=PERSIAN_CAT,
        epsilon=EPSILON,
        alpha=ALPHA,
        max_iterations=MAX_ITERATIONS
    )

    iterations_used.append(used_iterations)

    adv_pred, _ = predict_image(model, adv_image)

    if adv_pred == PERSIAN_CAT:
        success += 1

    count += 1

    print(
        f"{count:3d} | "
        f"iterations={used_iterations:3d} | "
        f"success={success}/{count}"
    )

    if count >= MAX_IMAGES:
        break

print("\n========== ITERATION TUNING ==========")
print(f"Images Tested      : {count}")
print(f"Successful Attacks : {success}")
print(f"Attack Success     : {success/count:.2%}")

print(f"\nHighest Iterations : {max(iterations_used)}")
print(f"Lowest Iterations  : {min(iterations_used)}")
print(f"Average Iterations : {sum(iterations_used)/len(iterations_used):.2f}")