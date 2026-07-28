import torch
from torchvision.transforms import ToPILImage
from pathlib import Path

to_pil = ToPILImage()


def create_perturbation(original, adversarial):
    """
    Returns a normalized perturbation tensor in [0,1]
    suitable for visualization.
    """

    perturbation = (adversarial - original).detach().cpu()

    perturbation = perturbation.abs()

    if perturbation.max() > 0:
        perturbation /= perturbation.max()

    return perturbation


def save_perturbation(original, adversarial, filename):
    """
    Save a perturbation visualization.
    """

    filename = Path(filename)
    filename.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    perturbation = create_perturbation(
        original,
        adversarial,
    )

    to_pil(perturbation).save(filename)