from datasets import load_dataset

def load_laion(
    dataset_name="copycat-project/laion2b6plus_dog",
    split="train"
):  
    dataset = load_dataset(dataset_name)
    return dataset[split]

