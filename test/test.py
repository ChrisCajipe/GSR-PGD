from dataset.laion_loader import load_laion

dataset = load_laion()

print(dataset)
print(dataset[0].keys())