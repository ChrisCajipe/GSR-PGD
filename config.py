from pathlib import Path

# ==========================
# MODE OF EXPERIMENTATION
# ==========================
MODE = "full"
# Options:
# "attack"   = generate adversarial images only
# "evaluate" = evaluate existing adversarial images only
# "full"     = generate + evaluate

# ==========================
# EVALUATION
# ==========================
EVALUATION = "tampered"
# Options
# "untampered" = all clean images
# "tampered" = all config-based attack tampered images 
# "50-50" = 50:50 config-based attack tampered images to clean images

# ==========================
# Attack
# ==========================
ATTACK = "gsr"
# "pgd" or "gsr"


# ==========================
# Dataset
# ==========================
MAX_IMAGES = 100
IMAGE_SIZE = (512, 512)

# ==========================
# Target Class
# ==========================
PERSIAN_CAT = 283

# ==========================
# PGD
# ==========================
EPSILON = 4 / 255      # PERTURBATION SIZE              (MOST OPTIMAL)
ALPHA = 2.55/255        # STEP SIZE                     (MOST OPTIMAL)
MAX_ITERATIONS = 10


# ==========================
# GSR
# ==========================
SIGMA = 0.03            # VARIANCE (STRENGTH OF GSR)    (MOST OPTIMAL)
LAMBDA = 0.5            # WEIGHT OF GSR                 (MOST OPTIMAL)


# ==========================
# Project Root
# ==========================
PROJECT_ROOT = Path(__file__).resolve().parent


# ==========================
# Results Directory
# ==========================

RESULTS_ROOT = PROJECT_ROOT / "results"


# permanent dataset
ORIGINAL_DIR = RESULTS_ROOT / "original"


# generated attacks
ATTACK_DIR = (
    RESULTS_ROOT
    / "attacks"
    / ATTACK
)

ADV_DIR = ATTACK_DIR / "adversarial"


# evaluation outputs
EVALUATION_DIR = (
    RESULTS_ROOT
    / "evaluations"
    / ATTACK
    / EVALUATION
)


LIGHTSHED_DIR = (
    EVALUATION_DIR
    / "lightshed"
)


TRUFOR_DIR = (
    EVALUATION_DIR
    / "trufor"
)


MIXED_DIR = (
    EVALUATION_DIR
    / "mixed"
)


# ==========================
# LightShed Repository
# ==========================

LIGHTSHED_ROOT = Path(
    r"C:\Users\Chris\Documents\LightShade-Artifact"
)


STANDARD_DETECTOR = (
    LIGHTSHED_ROOT
    / "lightshed"
    / "standard_detector.py"
)


CHECKPOINT = (
    LIGHTSHED_ROOT
    / "checkpoints"
    / "checkpointautoencoderAllRobustNewGlazeMistMetacloak_epoch_269.pth"
)

# ==========================
# TruFor
# ==========================

TRUFOR_ROOT = (
    PROJECT_ROOT
    / "TruFor-main"
)

TRUFOR_TEST = (
    TRUFOR_ROOT
    / "TruFor_train_test"
    / "test.py"
)

TRUFOR_CHECKPOINT = (
    TRUFOR_ROOT
    / "TruFor_train_test"
    / "pretrained_models"
    / "trufor.pth.tar"
)