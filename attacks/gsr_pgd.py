import torch
import torch.nn.functional as F
from utils.preprocessing import normalize_image
from attacks.spectral import spectral_loss

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# GAUSSIAN SPECTRAL REGULARIZED PGD
def gsr_pgd(                   
    model,
    image,
    target_label,
    epsilon,
    alpha,
    max_iterations,
    sigma,
    lambda_reg):    

    image = image.to(DEVICE)

    # initialize adversarial image
    adv_image = image.clone().detach().requires_grad_(True)                             

    for iteration in range(max_iterations):
        
        outputs = model(normalize_image(adv_image.unsqueeze(0)))
        target = torch.tensor([target_label], device=DEVICE, dtype=torch.long)
        
        # LOSS COMPUTATION
        classification_loss = F.cross_entropy(outputs, target)
        current_perturbation = adv_image - image
        spec_loss = spectral_loss(current_perturbation, sigma)
        loss = classification_loss + lambda_reg * spec_loss                              

        # GRADIENT COMPUTATION
        model.zero_grad()                                       
        loss.backward()

        # PGD UPDATE
        with torch.no_grad():
            adv_image = adv_image - alpha * adv_image.grad.sign()

            projected_perturbation = torch.clamp(
                adv_image - image,
                min=-epsilon,
                max=epsilon
            )

            adv_image = torch.clamp(image + projected_perturbation, 0, 1)

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