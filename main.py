from pipeline import generate_attacks, evaluate_defenses, print_defense_summary
from utils.create_folders import create_directories
from utils.dataset_builder import split_dataset
from dataset.laion_loader import load_laion
from config import MODE, ATTACK, TUNING_SIZE

create_directories()

print("STARTING...")

dataset = load_laion()

if MODE == "attack":
    count, success = generate_attacks(dataset=dataset)

elif MODE == "evaluate":
    resnet_results, quality_results, lightshed_results, trufor_results = evaluate_defenses()
    print_defense_summary(resnet_results, quality_results, lightshed_results, trufor_results)


elif MODE == "full":
    count, success = generate_attacks(dataset=dataset)
    resnet_results, quality_results, lightshed_results, trufor_results = evaluate_defenses()
    print_defense_summary(resnet_results, quality_results, lightshed_results, trufor_results)

else:
    raise ValueError(
        "MODE must be 'attack', 'evaluate', or 'full'"
    )