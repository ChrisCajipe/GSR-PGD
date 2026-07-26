from pipeline import generate_attacks, evaluate_defenses, print_defense_summary
from utils.create_folders import create_directories
from utils.dataset_builder import split_dataset
from dataset.laion_loader import load_laion
from config import MODE, ATTACK

create_directories()

dataset = load_laion()
tuning_indices, evaluation_indices = split_dataset(dataset,tuning_size=100)
evaluation_dataset = dataset.select(evaluation_indices)

if MODE == "attack":
    count, success = generate_attacks(dataset=evaluation_dataset)

elif MODE == "evaluate":
    resnet_results, quality_results, lightshed_results, trufor_results = evaluate_defenses()
    print_defense_summary(resnet_results, quality_results, lightshed_results, trufor_results)


elif MODE == "full":
    count, success = generate_attacks(dataset=evaluation_dataset)
    resnet_results, quality_results, lightshed_results, trufor_results = evaluate_defenses()
    print_defense_summary(resnet_results, quality_results, lightshed_results, trufor_results)

else:
    raise ValueError(
        "MODE must be 'attack', 'evaluate', or 'full'"
    )