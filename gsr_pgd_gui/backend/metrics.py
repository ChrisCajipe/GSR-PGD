import torch
import numpy as np

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


def tensor_to_numpy(image):
    """
    Converts torch image tensor:
    [C,H,W] -> [H,W,C]
    """

    image = image.detach().cpu()

    image = image.permute(1, 2, 0)

    image = image.numpy()

    image = np.clip(image, 0, 1)

    return image


def compute_psnr(original, compared):
    """
    Returns PSNR in dB
    """

    original_np = tensor_to_numpy(original)
    compared_np = tensor_to_numpy(compared)

    return peak_signal_noise_ratio(
        original_np,
        compared_np,
        data_range=1.0
    )


def compute_ssim(original, compared):
    """
    Returns SSIM score
    """

    original_np = tensor_to_numpy(original)
    compared_np = tensor_to_numpy(compared)

    return structural_similarity(
        original_np,
        compared_np,
        channel_axis=2,
        data_range=1.0
    )