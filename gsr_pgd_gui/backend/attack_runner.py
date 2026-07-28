import torch
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
)
from torchvision.utils import save_image
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.preprocessing import normalize_image
from attacks.pgd import targeted_pgd
from attacks.gsr_pgd import gsr_pgd
from backend.perturbation import (
    create_perturbation,
    save_perturbation,
)
from backend.classifier import predict_tensor

from backend.ui_config import (
    EPSILON,
    ALPHA,
    MAX_ITERATIONS,
    SIGMA,
    LAMBDA,

    ORIGINAL_DIR,

    PGD_ADV_DIR,
    GSR_ADV_DIR,

    ORIGINAL_FILENAME,
    PGD_FILENAME,
    GSR_FILENAME,

    PGD_LIGHTSHED_DIR,
    GSR_LIGHTSHED_DIR,

    PGD_TRUFOR_DIR,
    GSR_TRUFOR_DIR,

    PGD_PERT_DIR,
    GSR_PERT_DIR,

    PGD_PERT_FILENAME,
    GSR_PERT_FILENAME
)

from backend.lightshed_runner import run_lightshed
from backend.trufor_runner import run_trufor


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

weights = ResNet50_Weights.IMAGENET1K_V2

model = resnet50(
    weights=weights
).to(DEVICE)

model.eval()


def generate_attacks(
    image,
    target_class,
):
    """
    Generate both PGD and GSR-PGD attacks,
    evaluate them,
    and return everything needed by the GUI.
    """

    image = image.to(DEVICE)

    #
    # Generate attacks
    #

    pgd_image, pgd_iterations = targeted_pgd(
        model=model,
        image=image,
        target_label=target_class,
        epsilon=EPSILON,
        alpha=ALPHA,
        max_iterations=MAX_ITERATIONS,
    )

    gsr_image, gsr_iterations = gsr_pgd(
        model=model,
        image=image,
        target_label=target_class,
        epsilon=EPSILON,
        alpha=ALPHA,
        max_iterations=MAX_ITERATIONS,
        sigma=SIGMA,
        lambda_reg=LAMBDA,
    )

    pgd_prediction, pgd_conf = predict_tensor(pgd_image)
    gsr_prediction, gsr_conf = predict_tensor(gsr_image)

    print("PGD Prediction:", pgd_prediction, pgd_conf)
    print("GSR Prediction:", gsr_prediction, gsr_conf)

    pgd_perturbation = create_perturbation(
        image,
        pgd_image,
    )

    gsr_perturbation = create_perturbation(
        image,
        gsr_image,
    )

    #
    # Save images
    #

    save_image(
        image,
        ORIGINAL_DIR / ORIGINAL_FILENAME,
    )

    save_image(
        pgd_image,
        PGD_ADV_DIR / PGD_FILENAME,
    )

    save_image(
        gsr_image,
        GSR_ADV_DIR / GSR_FILENAME,
    )

    save_perturbation(
        image,
        pgd_image,
        PGD_PERT_DIR / PGD_PERT_FILENAME,
    )

    save_perturbation(
        image,
        gsr_image,
        GSR_PERT_DIR / GSR_PERT_FILENAME,
    )

    #
    # Evaluate with LightShed
    #

    pgd_lightshed_path = run_lightshed(
        PGD_ADV_DIR,
        PGD_LIGHTSHED_DIR,
    )

    gsr_lightshed_path = run_lightshed(
        GSR_ADV_DIR,
        GSR_LIGHTSHED_DIR,
    )

    pgd_lightshed = {
        "extracted_perturbation": pgd_lightshed_path
    }

    gsr_lightshed = {
        "extracted_perturbation": gsr_lightshed_path
    }

    #
    # Evaluate with TruFor
    #
    pgd_trufor = run_trufor(
        PGD_ADV_DIR,
        PGD_TRUFOR_DIR,
    )

    gsr_trufor = run_trufor(
        GSR_ADV_DIR,
        GSR_TRUFOR_DIR,
    )

    #
    # Return everything
    #

    # DEBUG
    with torch.no_grad():
        output = model(
            normalize_image(
                pgd_image.unsqueeze(0)
            )
        )

        prediction = output.argmax(dim=1).item()

    print(
        "Attack model prediction:",
        weights.meta["categories"][prediction]
    )

    print(
        "Target:",
        weights.meta["categories"][target_class]
    )

    return {

        "original": image,

        "pgd": pgd_image,
        "gsr": gsr_image,

        "pgd_iterations": pgd_iterations,
        "gsr_iterations": gsr_iterations,

        "pgd_perturbation": pgd_perturbation,
        "gsr_perturbation": gsr_perturbation,

        "pgd_lightshed": pgd_lightshed,
        "gsr_lightshed": gsr_lightshed,

        "pgd_trufor": pgd_trufor,
        "gsr_trufor": gsr_trufor,

        "target": target_class,
    }