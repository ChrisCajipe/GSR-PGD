from config import (
    ORIGINAL_DIR,
    ADV_DIR,
    LIGHTSHED_DIR,
    TRUFOR_DIR,
    MIXED_DIR
)

def create_directories():

    ORIGINAL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    ADV_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    LIGHTSHED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    TRUFOR_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    MIXED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )