import torch
import numpy as np

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)


def calculate_psnr(original, adversarial):

    original = (
        original
        .detach()
        .cpu()
        .numpy()
    )

    adversarial = (
        adversarial
        .detach()
        .cpu()
        .numpy()
    )


    # CHW -> HWC
    original = np.transpose(
        original,
        (1,2,0)
    )

    adversarial = np.transpose(
        adversarial,
        (1,2,0)
    )


    return peak_signal_noise_ratio(
        original,
        adversarial,
        data_range=1.0
    )



def calculate_ssim(original, adversarial):

    original = (
        original
        .detach()
        .cpu()
        .numpy()
    )

    adversarial = (
        adversarial
        .detach()
        .cpu()
        .numpy()
    )


    original = np.transpose(
        original,
        (1,2,0)
    )

    adversarial = np.transpose(
        adversarial,
        (1,2,0)
    )


    return structural_similarity(
        original,
        adversarial,
        channel_axis=2,
        data_range=1.0
    )



def evaluate_image_quality(original_dir, adv_dir):

    psnr_scores = []
    ssim_scores = []


    for adv_path in adv_dir.glob("*_adv.png"):

        image_id = (
            adv_path.stem
            .replace("_adv","")
        )


        original_path = (
            original_dir /
            f"{image_id}_original.png"
        )


        if not original_path.exists():
            continue


        from PIL import Image
        from torchvision.transforms import ToTensor


        original = ToTensor()(
            Image.open(original_path)
            .convert("RGB")
        )

        adversarial = ToTensor()(
            Image.open(adv_path)
            .convert("RGB")
        )


        psnr_scores.append(
            calculate_psnr(
                original,
                adversarial
            )
        )


        ssim_scores.append(
            calculate_ssim(
                original,
                adversarial
            )
        )


    return {
        "average_psnr":
            np.mean(psnr_scores),

        "average_ssim":
            np.mean(ssim_scores),

        "images":
            len(psnr_scores)
    }