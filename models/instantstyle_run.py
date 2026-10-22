from pathlib import Path
from PIL import Image
from diffusers import AutoPipelineForText2Image

import torch
import csv
import gc


# ============================================================
# MODEL SETTINGS
# ============================================================

BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_SUBFOLDER = "sdxl_models"
IP_ADAPTER_WEIGHT = "ip-adapter_sdxl.bin"


# ============================================================
# GENERATION SETTINGS
# ============================================================

PROMPTS = [
    (
        "prompt_01",
        "A small house beside a quiet lake."
    ),
    (
        "prompt_02",
        "A red bicycle parked beside a brick wall."
    ),
    (
        "prompt_03",
        "A bowl of oranges on a wooden table."
    ),
]

SEEDS = [
    42,
    1234
]

NEGATIVE_PROMPT = (
    "text, watermark, lowres, low quality, "
    "worst quality, deformed, glitch, "
    "low contrast, noisy, saturation, blurry"
)

GUIDANCE_SCALE = 5.0
NUM_INFERENCE_STEPS = 30

OUTPUT_WIDTH = 512
OUTPUT_HEIGHT = 512


# ============================================================
# INSTANTSTYLE CONFIGURATION
# ============================================================

STYLE_SCALE = {
    "up": {
        "block_0": [
            0.0,
            1.0,
            0.0
        ]
    }
}


# ============================================================
# HELPERS
# ============================================================

def get_reference_id(path, condition):

    stem = path.stem

    if condition == "clean":

        suffix = "_original"

        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]

    else:

        suffix = "_adv"

        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]

    return stem


def get_artist(reference_id):

    if "_art_" in reference_id:
        return reference_id.split("_art_")[0]

    return "unknown"


def find_references(folder, condition):

    if not folder.exists():
        raise FileNotFoundError(
            f"Reference folder not found: {folder}"
        )

    references = {}

    for path in sorted(folder.glob("*.png")):

        reference_id = get_reference_id(
            path,
            condition
        )

        references[reference_id] = path

    return references

# ============================================================
# MAIN INSTANTSTYLE FUNCTION
# ============================================================

def run_instantstyle(
    results_path="results/artwork_poc"
):

    results_path = Path(results_path)

    reference_folders = {
        "clean": (
            results_path
            / "pgd"
            / "original"
        ),

        "pgd": (
            results_path
            / "pgd"
            / "adversarial"
        ),

        "gsr_pgd": (
            results_path
            / "gsr"
            / "adversarial"
        ),
    }

    output_root = (
        results_path
        / "instantstyle"
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata_file = (
        output_root
        / "instantstyle_manifest.csv"
    )


    # ========================================================
    # FIND REFERENCE IMAGES
    # ========================================================

    print("\n" + "=" * 70)
    print("INSTANTSTYLE STYLE-MIMICRY GENERATION")
    print("=" * 70)

    reference_sets = {}

    for condition, folder in reference_folders.items():

        references = find_references(
            folder,
            condition
        )

        reference_sets[
            condition
        ] = references

        print(
            f"{condition.upper():8s}: "
            f"{len(references)} references"
        )


    # ========================================================
    # VERIFY MATCHING REFERENCES
    # ========================================================

    clean_ids = set(
        reference_sets["clean"].keys()
    )

    pgd_ids = set(
        reference_sets["pgd"].keys()
    )

    gsr_ids = set(
        reference_sets["gsr_pgd"].keys()
    )

    if not (
        clean_ids
        == pgd_ids
        == gsr_ids
    ):

        print("\nReference mismatch detected.")

        print(
            "Missing from PGD:",
            clean_ids - pgd_ids
        )

        print(
            "Missing from GSR-PGD:",
            clean_ids - gsr_ids
        )

        raise ValueError(
            "Clean, PGD, and GSR-PGD "
            "reference sets do not match."
        )


    reference_ids = sorted(
        clean_ids
    )


    # ========================================================
    # EXPECTED OUTPUT COUNT
    # ========================================================

    expected_outputs = (
        len(reference_ids)
        * 3
        * len(PROMPTS)
        * len(SEEDS)
    )

    print(
        f"\nMatched artworks   : "
        f"{len(reference_ids)}"
    )

    print(
        f"Expected generations: "
        f"{expected_outputs}"
    )


    # ========================================================
    # DEVICE
    # ========================================================

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA GPU not detected. "
            "SDXL + InstantStyle is not practical "
            "for this experiment on CPU."
        )


    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )


    # ========================================================
    # LOAD SDXL
    # ========================================================

    print("\nLoading SDXL...")

    pipe = (
        AutoPipelineForText2Image
        .from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            use_safetensors=True,
            low_cpu_mem_usage=True,
        )
    )


    # ========================================================
    # LOAD IP-ADAPTER
    # ========================================================

    print("Loading IP-Adapter...")

    pipe.load_ip_adapter(
        IP_ADAPTER_REPO,
        subfolder=IP_ADAPTER_SUBFOLDER,
        weight_name=IP_ADAPTER_WEIGHT,
    )


    # Style-only InstantStyle configuration
    pipe.set_ip_adapter_scale(
        STYLE_SCALE
    )


    # ========================================================
    # MEMORY OPTIMIZATION
    # ========================================================

    # Reduce VAE memory usage
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

    # Important for low-VRAM GPUs
    pipe.enable_model_cpu_offload()

    print("\nInstantStyle model loaded.")


    # ========================================================
    # GENERATION
    # ========================================================

    manifest_rows = []

    generation_number = 0

    def is_valid_image(path):
        if not path.exists():
            return False

        try:
            with Image.open(path) as image:
                image.verify()

            return True

        except Exception:
            return False
        
    for reference_id in reference_ids:

        artist = get_artist(
            reference_id
        )

        for condition in [
            "clean",
            "pgd",
            "gsr_pgd"
        ]:

            reference_path = (
                reference_sets[
                    condition
                ][reference_id]
            )

            style_image = (
                Image
                .open(reference_path)
                .convert("RGB")
            )

            style_image = style_image.resize(
                (512, 512),
                Image.Resampling.LANCZOS
            )


            for prompt_id, prompt in PROMPTS:

                for seed in SEEDS:

                    generation_number += 1

                    output_dir = (
                        output_root
                        / condition
                        / artist
                        / reference_id
                    )

                    output_dir.mkdir(
                        parents=True,
                        exist_ok=True
                    )

                    output_filename = (
                        f"{prompt_id}"
                        f"_seed_{seed}.png"
                    )

                    output_path = (
                        output_dir
                        / output_filename
                    )


                    print(
                        f"\n"
                        f"[{generation_number}/"
                        f"{expected_outputs}] "
                        f"{condition.upper()} | "
                        f"{artist} | "
                        f"{prompt_id} | "
                        f"seed {seed}"
                    )


                    # ----------------------------------------
                    # DON'T REGENERATE EXISTING FILES
                    # ----------------------------------------

                    if is_valid_image(output_path):
                        print("Already exists and is valid. Skipping.")

                    else:
                        if output_path.exists():

                            print(
                                "Incomplete/corrupted image found. "
                                "Regenerating..."
                            )

                            output_path.unlink()

                        generator = (
                            torch.Generator(
                                device="cpu"
                            )
                            .manual_seed(seed)
                        )

                        with torch.inference_mode():

                            result = pipe(
                                prompt=prompt,

                                negative_prompt=NEGATIVE_PROMPT,

                                ip_adapter_image=style_image,

                                guidance_scale=GUIDANCE_SCALE,

                                num_inference_steps=NUM_INFERENCE_STEPS,

                                generator=generator,

                                height=OUTPUT_HEIGHT,

                                width=OUTPUT_WIDTH,
                            )

                        generated_image = result.images[0]

                        generated_image.save(
                            output_path
                        )

                        print(
                            f"Saved: {output_path}"
                        )



                    manifest_rows.append({

                        "artist":
                            artist,

                        "reference_id":
                            reference_id,

                        "condition":
                            condition,

                        "reference_file":
                            str(reference_path),

                        "prompt_id":
                            prompt_id,

                        "prompt":
                            prompt,

                        "seed":
                            seed,

                        "guidance_scale":
                            GUIDANCE_SCALE,

                        "num_inference_steps":
                            NUM_INFERENCE_STEPS,

                        "output_width":
                            OUTPUT_WIDTH,

                        "output_height":
                            OUTPUT_HEIGHT,

                        "output_file":
                            str(output_path),
                    })


                    gc.collect()

                    torch.cuda.empty_cache()


        
    # ========================================================
    # SAVE MANIFEST
    # ========================================================

    fieldnames = [
        "artist",
        "reference_id",
        "condition",
        "reference_file",
        "prompt_id",
        "prompt",
        "seed",
        "guidance_scale",
        "num_inference_steps",
        "output_width",
        "output_height",
        "output_file",
    ]

    with open(
        metadata_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            manifest_rows
        )


    print("\n" + "=" * 70)
    print("INSTANTSTYLE GENERATION COMPLETE")
    print("=" * 70)

    print(
        f"Outputs expected : {expected_outputs}"
    )

    print(
        f"Output directory : {output_root}"
    )

    print(
        f"Manifest         : {metadata_file}"
    )

    return {
        "artworks": len(reference_ids),
        "generated": expected_outputs,
        "output_dir": output_root,
        "manifest": metadata_file,
    }


# ============================================================
# ALLOW DIRECT EXECUTION TOO
# ============================================================

if __name__ == "__main__":

    run_instantstyle()


