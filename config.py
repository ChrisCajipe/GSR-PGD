from pathlib import Path

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
# Attack
# ==========================
ATTACK = "pgd"   # "pgd" or "gsr"


# ==========================
# PGD
# ==========================
EPSILON = 8 / 255
ALPHA = 2 / 255
MAX_ITERATIONS = 10


# ==========================
# GSR
# ==========================
SIGMA = 0.05
LAMBDA = 0.1


# ==========================
# Project Root
# ==========================
PROJECT_ROOT = Path(__file__).resolve().parent


# ==========================
# Results Directory
# ==========================

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / f"{ATTACK}-results"
)

ORIGINAL_DIR = RESULTS_DIR / "original"

ADV_DIR = RESULTS_DIR / "adversarial"

LIGHTSHED_DIR = RESULTS_DIR / "lightshed_results"



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
    / "trufor"
    / "TruFor_train_test"
)

TRUFOR_CHECKPOINT = (
    TRUFOR_ROOT
    / "weights"
    / "trufor_ph3"
    / "best.pth.tar"
)