import torch

def spectral_loss(perturbation, sigma=1.0):
    # GAUSSIAN NOISE GENERATION
    gaussian = torch.randn_like(perturbation) * sigma

    # FFT COMPUTATION
    delta_fft = torch.fft.fftn(
        perturbation,
        dim=(-2, -1)
    )

    gaussian_fft = torch.fft.fftn(
        gaussian,
        dim=(-2, -1)
    )

    # MAGNITUDE SPECTRA
    delta_mag = torch.abs(delta_fft)
    gaussian_mag = torch.abs(gaussian_fft)

    loss = torch.mean(
        (delta_mag - gaussian_mag) ** 2
    )

    return loss