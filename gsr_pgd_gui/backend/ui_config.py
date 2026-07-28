from pathlib import Path

#
# PROJECT
#

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

#
# LIGHTSHED
#

LIGHTSHED_CODE_DIR = (
    PROJECT_ROOT
    / "lightshade-artifact"
    / "lightshed"
)

LIGHTSHED_SCRIPT = (
    LIGHTSHED_CODE_DIR
    / "standard_detector.py"
)

#
# TRUFOR
#

TRUFOR_DIR = (
    PROJECT_ROOT
    / "TruFor-main"
    / "TruFor_train_test"
)

TRUFOR_TEST = (
    TRUFOR_DIR
    / "test.py"
)

TRUFOR_CHECKPOINT = (
    TRUFOR_DIR
    / "pretrained_models"
    / "trufor.pth.tar"
)

# 
# PGD
# 
EPSILON = 4 / 255      # PERTURBATION SIZE              (MOST OPTIMAL)
ALPHA = 0.01            # STEP SIZE                     (MOST OPTIMAL)
MAX_ITERATIONS = 10


# 
# GSR
# 
SIGMA = 0.03            # VARIANCE (STRENGTH OF GSR)    (MOST OPTIMAL)
LAMBDA = 0.75            # WEIGHT OF GSR                 (MOST OPTIMAL)

#
# MODEL
#

TARGET_CLASSES = {
    "European fire salamander": 25,
    "Sulphur-crested cockatoo": 89,
    "Wallaby": 104,
    "Slug": 114,
    "Dowitcher": 142,
    "Komondor": 228,
    "Siberian husky": 250,
    "Tabby cat": 281,
    "Persian cat": 283,
    "Flute": 558,
    "Minibus": 654,
    "Packet": 692,
    "Radio": 754,
    "Reel": 758,
    "Reflex Camera": 759,
    "Wreck": 913,
}

#
# RESULTS
#

RESULTS_ROOT = PROJECT_ROOT / "gsr_pgd_gui" / "prototype_results"

#
# ORIGINAL
#

ORIGINAL_DIR = RESULTS_ROOT / "original"

#
# PGD
#

PGD_DIR = RESULTS_ROOT / "attacks" / "pgd"

PGD_ADV_DIR = PGD_DIR / "adversarial"
PGD_PERT_DIR = PGD_DIR / "perturbation"

#
# GSR-PGD
#

GSR_DIR = RESULTS_ROOT / "attacks" / "gsr"

GSR_ADV_DIR = GSR_DIR / "adversarial"
GSR_PERT_DIR = GSR_DIR / "perturbation"

#
# EVALUATIONS
#

EVALUATION_DIR = RESULTS_ROOT / "evaluations"

#
# PGD EVALUATION
#

PGD_LIGHTSHED_DIR = EVALUATION_DIR / "pgd" / "lightshed"
PGD_TRUFOR_DIR = EVALUATION_DIR / "pgd" / "trufor"

#
# GSR-PGD EVALUATION
#

GSR_LIGHTSHED_DIR = EVALUATION_DIR / "gsr" / "lightshed"
GSR_TRUFOR_DIR = EVALUATION_DIR / "gsr" / "trufor"

#
# CREATE DIRECTORIES
#

DIRECTORIES = [
    ORIGINAL_DIR,

    PGD_ADV_DIR,
    PGD_PERT_DIR,

    GSR_ADV_DIR,
    GSR_PERT_DIR,

    PGD_LIGHTSHED_DIR,
    PGD_TRUFOR_DIR,

    GSR_LIGHTSHED_DIR,
    GSR_TRUFOR_DIR,
]

for directory in DIRECTORIES:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

ORIGINAL_FILENAME = "original.png"

PGD_FILENAME = "pgd.png"
GSR_FILENAME = "gsr.png"

PGD_PERT_FILENAME = "pgd_perturbation.png"
GSR_PERT_FILENAME = "gsr_perturbation.png"

