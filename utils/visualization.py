import torch
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt

to_pil = ToPILImage()


def save_image(image, filename):
    """
    Save a tensor image in [0,1] as PNG.
    """

    image = image.detach().cpu().clamp(0, 1)

    pil = to_pil(image)
    pil.save(filename)


def show_images(original, adversarial):
    """
    Display original and adversarial images side-by-side.
    """

    original = original.detach().cpu().clamp(0, 1)
    adversarial = adversarial.detach().cpu().clamp(0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(to_pil(original))
    axes[0].set_title("Original")

    axes[1].imshow(to_pil(adversarial))
    axes[1].set_title("Adversarial")

    plt.show()


def show_perturbation(original, adversarial):
    """
    Visualize the perturbation.
    """

    perturbation = (adversarial - original).detach().cpu()

    perturbation = perturbation.abs()
    perturbation /= perturbation.max()

    plt.imshow(to_pil(perturbation))
    plt.title("Perturbation")
    plt.axis("off")
    plt.show()