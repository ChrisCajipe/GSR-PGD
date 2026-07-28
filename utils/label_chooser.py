import random

# Set a seed for reproducibility
SEED = 42
random.seed(SEED)

selected_numbers = sorted(random.sample(range(1000), 15))

print(selected_numbers)