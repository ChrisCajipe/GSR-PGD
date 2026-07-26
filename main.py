from pipeline import generate_attacks, evaluate_defenses, print_defense_summary
from utils.create_folders import create_directories
from config import MODE, ATTACK

create_directories()

if MODE == "attack":
    count, success = generate_attacks()

elif MODE == "evaluate":
    resnet_results, quality_results, lightshed_results, trufor_results = evaluate_defenses()
    print_defense_summary(resnet_results, quality_results, lightshed_results, trufor_results)


elif MODE == "full":
    count, success = generate_attacks()
    resnet_results, quality_results, lightshed_results, trufor_results = evaluate_defenses()
    print_defense_summary(resnet_results, quality_results, lightshed_results, trufor_results)

else:
    raise ValueError(
        "MODE must be 'attack', 'evaluate', or 'full'"
    )