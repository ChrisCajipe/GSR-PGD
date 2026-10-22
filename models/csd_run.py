from pathlib import Path
from collections import defaultdict
from statistics import mean, stdev

import csv
import sys
import torch

from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF

from huggingface_hub import hf_hub_download


# ============================================================
# CSD IMPLEMENTATION
# ============================================================

CSD_SCORE_DIR = Path(
    "external/CSD_Score"
).resolve()

if not CSD_SCORE_DIR.exists():
    raise FileNotFoundError(
        "external/CSD_Score was not found.\n"
        "Clone it using:\n"
        "git clone https://github.com/haofanwang/"
        "CSD_Score.git external/CSD_Score"
    )

sys.path.insert(
    0,
    str(CSD_SCORE_DIR)
)

from model import (
    CSD_CLIP,
    convert_state_dict,
)


# ============================================================
# PATHS
# ============================================================

RESULTS_ROOT = Path(
    "results/artwork_poc"
)

CLEAN_REFERENCE_DIR = (
    RESULTS_ROOT
    / "pgd"
    / "original"
)

INSTANTSTYLE_DIR = (
    RESULTS_ROOT
    / "instantstyle"
)

MANIFEST_FILE = (
    INSTANTSTYLE_DIR
    / "instantstyle_manifest.csv"
)

CSD_RESULTS_DIR = (
    RESULTS_ROOT
    / "csd"
)

CSD_RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


DETAILED_FILE = (
    CSD_RESULTS_DIR
    / "csd_generated_scores.csv"
)

ARTWORK_SUMMARY_FILE = (
    CSD_RESULTS_DIR
    / "csd_artwork_summary.csv"
)

ARTIST_SUMMARY_FILE = (
    CSD_RESULTS_DIR
    / "csd_artist_summary.csv"
)

CONDITION_SUMMARY_FILE = (
    CSD_RESULTS_DIR
    / "csd_condition_summary.csv"
)


# ============================================================
# MODEL SETTINGS
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

CSD_MODEL_REPO = (
    "tomg-group-umd/CSD-ViT-L"
)

CSD_MODEL_FILE = (
    "pytorch_model.bin"
)


# ============================================================
# PREPROCESSING
# ============================================================

# These are the CLIP normalization values
# used by the CSD implementation.

normalize = transforms.Normalize(
    mean=(
        0.48145466,
        0.4578275,
        0.40821073,
    ),
    std=(
        0.26862954,
        0.26130258,
        0.27577711,
    ),
)


preprocess = transforms.Compose([
    transforms.Resize(
        size=224,
        interpolation=TF.InterpolationMode.BICUBIC,
    ),

    transforms.CenterCrop(
        224
    ),

    transforms.ToTensor(),

    normalize,
])


# ============================================================
# LOAD CSD MODEL
# ============================================================

def load_csd_model():

    print("=" * 70)
    print("LOADING CSD MODEL")
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    # --------------------------------------------------------
    # Initialize CSD architecture
    # --------------------------------------------------------

    # Passing ViT-L/14 lets OpenAI CLIP retrieve the
    # corresponding backbone if it is not already cached.

    model = CSD_CLIP(
        "vit_large",
        "default",
        model_path="ViT-L/14",
    )


    # --------------------------------------------------------
    # Download official CSD checkpoint
    # --------------------------------------------------------

    print(
        "\nRetrieving CSD ViT-L checkpoint..."
    )

    model_path = hf_hub_download(
        repo_id=CSD_MODEL_REPO,
        filename=CSD_MODEL_FILE,
    )


    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    # The official checkpoint is a PyTorch checkpoint rather
    # than safetensors.
    #
    # weights_only=False is required by some newer PyTorch
    # versions for this particular checkpoint format.
    #
    # Only do this with the trusted official checkpoint above.

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )


    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        state_dict = checkpoint


    state_dict = convert_state_dict(
        state_dict
    )


    missing, unexpected = (
        model.load_state_dict(
            state_dict,
            strict=False,
        )
    )


    if missing:
        print(
            f"Missing keys: {len(missing)}"
        )

    if unexpected:
        print(
            f"Unexpected keys: {len(unexpected)}"
        )


    model = model.to(
        DEVICE
    )

    model.eval()


    print(
        "\nCSD model loaded."
    )

    return model


# ============================================================
# EXTRACT CSD STYLE EMBEDDING
# ============================================================

@torch.no_grad()
def extract_style_embedding(
    model,
    image_path,
):

    image = (
        Image
        .open(image_path)
        .convert("RGB")
    )


    image_tensor = (
        preprocess(image)
        .unsqueeze(0)
        .to(DEVICE)
    )


    # CSD returns:
    #
    # feature
    # content_output
    # style_output
    #
    # We specifically need style_output.

    _, _, style_output = model(
        image_tensor
    )


    # style_output is already L2-normalized
    # by the CSD model.

    embedding = (
        style_output[0]
        .detach()
        .cpu()
    )


    return embedding


# ============================================================
# LOAD INSTANTSTYLE MANIFEST
# ============================================================

def load_manifest():

    if not MANIFEST_FILE.exists():

        raise FileNotFoundError(
            f"Manifest not found: "
            f"{MANIFEST_FILE}"
        )


    with open(
        MANIFEST_FILE,
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        rows = list(
            csv.DictReader(file)
        )


    if not rows:

        raise ValueError(
            "InstantStyle manifest is empty."
        )


    print(
        f"\nManifest entries: "
        f"{len(rows)}"
    )


    return rows


# ============================================================
# GET CLEAN REFERENCES
# ============================================================

def load_clean_reference_paths(
    required_reference_ids,
):

    references = {}


    for path in sorted(
        CLEAN_REFERENCE_DIR.glob(
            "*_original.png"
        )
    ):

        stem = path.stem


        suffix = "_original"

        if stem.endswith(suffix):

            reference_id = stem[
                :-len(suffix)
            ]

        else:

            continue


        if (
            reference_id
            in required_reference_ids
        ):

            references[
                reference_id
            ] = path


    missing = (
        required_reference_ids
        - set(references.keys())
    )


    if missing:

        raise ValueError(
            "Missing clean references:\n"
            + "\n".join(
                sorted(missing)
            )
        )


    return references


# ============================================================
# HELPER
# ============================================================

def get_artist(
    reference_id
):

    if "_art_" not in reference_id:

        raise ValueError(
            f"Unable to determine artist "
            f"from ID: {reference_id}"
        )


    return reference_id.split(
        "_art_"
    )[0]


# ============================================================
# CSD EVALUATION
# ============================================================

def run_csd_evaluation():

    print("\n" + "=" * 70)
    print("CSD STYLE-SIMILARITY EVALUATION")
    print("=" * 70)


    model = load_csd_model()

    manifest_rows = load_manifest()


    # ========================================================
    # VERIFY GENERATED OUTPUTS
    # ========================================================

    missing_outputs = []


    for row in manifest_rows:

        output_path = Path(
            row["output_file"]
        )

        if not output_path.exists():

            missing_outputs.append(
                str(output_path)
            )


    if missing_outputs:

        raise FileNotFoundError(
            f"{len(missing_outputs)} "
            "generated images are missing."
        )


    print(
        f"Verified generated images: "
        f"{len(manifest_rows)}"
    )


    # ========================================================
    # FIND SOURCE ARTWORK IDS
    # ========================================================

    reference_ids = {
        row["reference_id"]
        for row in manifest_rows
    }


    print(
        f"Unique source artworks: "
        f"{len(reference_ids)}"
    )


    clean_paths = (
        load_clean_reference_paths(
            reference_ids
        )
    )


    # ========================================================
    # GROUP SOURCE ARTWORKS BY ARTIST
    # ========================================================

    artist_references = defaultdict(
        list
    )


    for reference_id in sorted(
        reference_ids
    ):

        artist = get_artist(
            reference_id
        )

        artist_references[
            artist
        ].append(
            reference_id
        )


    print(
        "\nReference artworks per artist:"
    )


    for artist, refs in sorted(
        artist_references.items()
    ):

        print(
            f"  {artist}: {len(refs)}"
        )


        if len(refs) != 5:

            raise ValueError(
                f"{artist} should have "
                f"5 artworks, but has "
                f"{len(refs)}."
            )


    # ========================================================
    # EMBED ALL 15 CLEAN ARTWORKS ONCE
    # ========================================================

    print(
        "\nExtracting CSD embeddings "
        "for clean artworks..."
    )


    clean_embeddings = {}


    for number, reference_id in enumerate(
        sorted(reference_ids),
        start=1,
    ):

        print(
            f"[{number}/"
            f"{len(reference_ids)}] "
            f"{reference_id}"
        )


        clean_embeddings[
            reference_id
        ] = extract_style_embedding(
            model,
            clean_paths[
                reference_id
            ],
        )


    # ========================================================
    # SCORE ALL GENERATED IMAGES
    # ========================================================

    print(
        "\nScoring generated images..."
    )


    detailed_rows = []


    total = len(
        manifest_rows
    )


    for index, row in enumerate(
        manifest_rows,
        start=1,
    ):

        reference_id = row[
            "reference_id"
        ]

        artist = row[
            "artist"
        ]

        condition = row[
            "condition"
        ]

        output_path = Path(
            row["output_file"]
        )


        print(
            f"[{index}/{total}] "
            f"{condition.upper()} | "
            f"{reference_id} | "
            f"{row['prompt_id']} | "
            f"seed {row['seed']}"
        )


        generated_embedding = (
            extract_style_embedding(
                model,
                output_path,
            )
        )


        # ----------------------------------------------------
        # LEAVE-ONE-OUT ARTIST REFERENCES
        # ----------------------------------------------------

        comparison_ids = [

            candidate_id

            for candidate_id
            in artist_references[
                artist
            ]

            if candidate_id
            != reference_id
        ]


        if len(comparison_ids) != 4:

            raise ValueError(
                f"Expected 4 leave-one-out "
                f"references for "
                f"{reference_id}, but found "
                f"{len(comparison_ids)}."
            )


        similarities = []


        for comparison_id in comparison_ids:

            reference_embedding = (
                clean_embeddings[
                    comparison_id
                ]
            )


            # Both CSD style embeddings are
            # L2 normalized.
            #
            # Therefore dot product =
            # cosine similarity.

            similarity = torch.dot(
                generated_embedding,
                reference_embedding,
            ).item()


            similarities.append(
                similarity
            )


        mean_csd = mean(
            similarities
        )


        detailed_rows.append({

            "artist":
                artist,

            "reference_id":
                reference_id,

            "condition":
                condition,

            "prompt_id":
                row["prompt_id"],

            "prompt":
                row["prompt"],

            "seed":
                row["seed"],

            "generated_file":
                str(output_path),

            "comparison_reference_1":
                comparison_ids[0],

            "csd_similarity_1":
                similarities[0],

            "comparison_reference_2":
                comparison_ids[1],

            "csd_similarity_2":
                similarities[1],

            "comparison_reference_3":
                comparison_ids[2],

            "csd_similarity_3":
                similarities[2],

            "comparison_reference_4":
                comparison_ids[3],

            "csd_similarity_4":
                similarities[3],

            "mean_csd_similarity":
                mean_csd,
        })


    # ========================================================
    # SAVE 270-ROW DETAILED RESULTS
    # ========================================================

    detailed_fields = [
        "artist",
        "reference_id",
        "condition",
        "prompt_id",
        "prompt",
        "seed",
        "generated_file",

        "comparison_reference_1",
        "csd_similarity_1",

        "comparison_reference_2",
        "csd_similarity_2",

        "comparison_reference_3",
        "csd_similarity_3",

        "comparison_reference_4",
        "csd_similarity_4",

        "mean_csd_similarity",
    ]


    with open(
        DETAILED_FILE,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=detailed_fields,
        )

        writer.writeheader()

        writer.writerows(
            detailed_rows
        )


    # ========================================================
    # ARTWORK-LEVEL AGGREGATION
    #
    # 3 prompts × 2 seeds = 6 scores
    # per artwork per condition
    # ========================================================

    artwork_groups = defaultdict(
        list
    )


    for row in detailed_rows:

        key = (
            row["artist"],
            row["reference_id"],
            row["condition"],
        )

        artwork_groups[
            key
        ].append(
            row[
                "mean_csd_similarity"
            ]
        )


    artwork_summary = []


    for (
        artist,
        reference_id,
        condition,
    ), scores in sorted(
        artwork_groups.items()
    ):

        artwork_summary.append({

            "artist":
                artist,

            "reference_id":
                reference_id,

            "condition":
                condition,

            "n_generations":
                len(scores),

            "mean_csd_similarity":
                mean(scores),

            "sd_csd_similarity":
                (
                    stdev(scores)
                    if len(scores) > 1
                    else 0.0
                ),
        })


    with open(
        ARTWORK_SUMMARY_FILE,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        fields = [
            "artist",
            "reference_id",
            "condition",
            "n_generations",
            "mean_csd_similarity",
            "sd_csd_similarity",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(
            artwork_summary
        )


    # ========================================================
    # ARTIST × CONDITION SUMMARY
    # ========================================================

    artist_groups = defaultdict(
        list
    )


    for row in artwork_summary:

        key = (
            row["artist"],
            row["condition"],
        )

        artist_groups[
            key
        ].append(
            row[
                "mean_csd_similarity"
            ]
        )


    artist_summary = []


    for (
        artist,
        condition,
    ), scores in sorted(
        artist_groups.items()
    ):

        artist_summary.append({

            "artist":
                artist,

            "condition":
                condition,

            "n_artworks":
                len(scores),

            "mean_csd_similarity":
                mean(scores),

            "sd_csd_similarity":
                (
                    stdev(scores)
                    if len(scores) > 1
                    else 0.0
                ),
        })


    with open(
        ARTIST_SUMMARY_FILE,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        fields = [
            "artist",
            "condition",
            "n_artworks",
            "mean_csd_similarity",
            "sd_csd_similarity",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(
            artist_summary
        )


    # ========================================================
    # OVERALL CONDITION SUMMARY
    #
    # IMPORTANT:
    # Uses 15 ARTWORK-LEVEL scores,
    # not 90 generations as independent samples.
    # ========================================================

    condition_groups = defaultdict(
        list
    )


    for row in artwork_summary:

        condition_groups[
            row["condition"]
        ].append(
            row[
                "mean_csd_similarity"
            ]
        )


    condition_summary = []


    for condition in [
        "clean",
        "pgd",
        "gsr_pgd",
    ]:

        scores = (
            condition_groups[
                condition
            ]
        )


        condition_summary.append({

            "condition":
                condition,

            "n_artworks":
                len(scores),

            "mean_csd_similarity":
                mean(scores),

            "sd_csd_similarity":
                stdev(scores),
        })


    with open(
        CONDITION_SUMMARY_FILE,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        fields = [
            "condition",
            "n_artworks",
            "mean_csd_similarity",
            "sd_csd_similarity",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(
            condition_summary
        )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "CSD EVALUATION COMPLETE"
    )

    print(
        "=" * 70
    )


    for row in condition_summary:

        print(
            f"\n"
            f"{row['condition'].upper()}"
        )

        print(
            f"  Artworks : "
            f"{row['n_artworks']}"
        )

        print(
            f"  Mean CSD : "
            f"{row['mean_csd_similarity']:.6f}"
        )

        print(
            f"  SD       : "
            f"{row['sd_csd_similarity']:.6f}"
        )


    print(
        "\nFiles:"
    )

    print(
        f"Detailed scores : "
        f"{DETAILED_FILE}"
    )

    print(
        f"Artwork summary : "
        f"{ARTWORK_SUMMARY_FILE}"
    )

    print(
        f"Artist summary  : "
        f"{ARTIST_SUMMARY_FILE}"
    )

    print(
        f"Condition summary: "
        f"{CONDITION_SUMMARY_FILE}"
    )


    return condition_summary


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    run_csd_evaluation()