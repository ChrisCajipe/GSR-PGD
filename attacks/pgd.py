import torch
import torch.nn.functional as F
from utils.preprocessing import normalize_image

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# STANDARD PGD
def targeted_pgd(                   
    model,
    image,
    target_label,
    epsilon,
    alpha,
    max_iterations):    

    image = image.to(DEVICE)
    
    # INITIALIZE ADVERSARIAL IMAGE
    adv_image = image.clone().detach().requires_grad_(True)                             

    for iteration in range(max_iterations):
        
        outputs = model(normalize_image(adv_image.unsqueeze(0)))
        target = torch.tensor([target_label], device=image.device, dtype=torch.long)
        
        # LOSS COMPUTATION
        loss = F.cross_entropy(outputs, target)                                 

        # GRADIENT COMPUTATION
        model.zero_grad()                                       
        loss.backward()

        # PGD UPDATE
        with torch.no_grad():
            adv_image = adv_image - alpha * adv_image.grad.sign()

            perturbation = torch.clamp(
                adv_image - image,
                min=-epsilon,
                max=epsilon
            )

            adv_image = torch.clamp(
                image + perturbation,
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
    
    
    