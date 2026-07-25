import statistics
import torch
import torch.nn.functional as F

from dataset.laion_loader import load_laion
from utils.preprocessing import preprocess_image, normalize_image
from utils.imagenet_labels import is_dog
from attacks.spectral import spectral_loss
from models.resnet import load_resnet, predict_image


# ==========================================================
# SETTINGS
# ==========================================================

PERSIAN_CAT = 283

MAX_IMAGES = 100
MAX_ITERATIONS = 40

EPSILON = 8 / 255
ALPHA = 2 / 255

SIGMA = 0.05
LAMBDA = 0.1


# ==========================================================
# GSR-PGD (Pilot Version)
# ==========================================================

def gsr_pgd(
    model,
    image,
    target_label,
    epsilon=EPSILON,
    alpha=ALPHA,
    max_iterations=MAX_ITERATIONS,
    sigma=SIGMA,
    lambda_reg=LAMBDA,
):

    adv_image = image.clone().detach().requires_grad_(True)

    for iteration in range(max_iterations):

        outputs = model(normalize_image(adv_image.unsqueeze(0)))
        target = torch.tensor(
            [target_label],
            device=image.device,
            dtype=torch.long
        )

        classification_loss = F.cross_entropy(outputs, target)

        perturbation = adv_image - image

        spec_loss = spectral_loss(
            perturbation,
            sigma
        )

        loss = classification_loss + lambda_reg * spec_loss

        model.zero_grad()
        loss.backward()

        with torch.no_grad():

            adv_image = adv_image - alpha * adv_image.grad.sign()

            projected = torch.clamp(
                adv_image - image,
                min=-epsilon,
                max=epsilon
            )

            adv_image = torch.clamp(
                image + projected,
                0,
                1
            )

            prediction = model(
                normalize_image(
                    adv_image.unsqueeze(0)
                )
            )

            predicted_class = prediction.argmax(dim=1).item()

            if predicted_class == target_label:
                return adv_image.detach(), iteration + 1

        adv_image = adv_image.detach().requires_grad_(True)

    return adv_image.detach(), max_iterations


# ==========================================================
# MAIN
# ==========================================================

dataset = load_laion()
model = load_resnet()

tested = 0
success = 0

iteration_counts = []

print("=" * 60)
print("GSR-PGD PILOT")
print("=" * 60)

for sample in dataset:

    image = preprocess_image(sample["image"])

    prediction_id, _ = predict_image(
        model,
        image
    )

    if not is_dog(prediction_id):
        continue

    adv_image, iterations = gsr_pgd(
        model=model,
        image=image,
        target_label=PERSIAN_CAT,
    )

    adv_prediction_id, _ = predict_image(
        model,
        adv_image
    )

    tested += 1

    if adv_prediction_id == PERSIAN_CAT:
        success += 1

    iteration_counts.append(iterations)

    print(
        f"[{tested:03d}/{MAX_IMAGES}] "
        f"{sample['image_id']} "
        f"| Iterations: {iterations:2d} "
        f"| {'SUCCESS' if adv_prediction_id == PERSIAN_CAT else 'FAILED'}"
    )

    if tested >= MAX_IMAGES:
        break


print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"Images Tested      : {tested}")
print(f"Successful Attacks : {success}")
print(f"Attack Success Rate: {100*success/tested:.2f}%")

print(f"Average Iteration  : {statistics.mean(iteration_counts):.2f}")
print(f"Highest Iteration  : {max(iteration_counts)}")
print(f"Lowest Iteration   : {min(iteration_counts)}")