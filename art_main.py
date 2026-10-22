from pathlib import Path
from models.instantstyle_run import run_instantstyle
from pipeline import generate_attacks
from dataset.artwork_loader import load_artworks


# ============================================================
# SETTINGS
# ============================================================

dataset_path = "dataset/wikiart_poc"

results_path = Path(
    "results/artwork_poc"
)


# ============================================================
# LOAD ARTWORK DATA
# ============================================================

print("=" * 70)
print("GSR-PGD ARTWORK PROOF-OF-CONCEPT")
print("=" * 70)

dataset = load_artworks(
    dataset_path=dataset_path
)

total_images = len(dataset)

print(
    f"\nTotal artwork images: {total_images}"
)


# ============================================================
# PGD
# ============================================================

print("\n" + "=" * 70)
print("GENERATING PGD ARTWORK ATTACKS")
print("=" * 70)

pgd_original_dir = (
    results_path
    / "pgd"
    / "original"
)

pgd_adv_dir = (
    results_path
    / "pgd"
    / "adversarial"
)

pgd_count, pgd_success, pgd_time = generate_attacks(
    dataset=dataset,

    attack="pgd",

    total_images=total_images,

    # No tuning/evaluation split for this PoC
    create_split=False,

    original_dir=pgd_original_dir,
    adv_dir=pgd_adv_dir,
)


# ============================================================
# GSR-PGD
# ============================================================

print("\n" + "=" * 70)
print("GENERATING GSR-PGD ARTWORK ATTACKS")
print("=" * 70)

gsr_original_dir = (
    results_path
    / "gsr"
    / "original"
)

gsr_adv_dir = (
    results_path
    / "gsr"
    / "adversarial"
)

gsr_count, gsr_success, gsr_time = generate_attacks(
    dataset=dataset,

    attack="gsr",

    total_images=total_images,

    # No tuning/evaluation split for this PoC
    create_split=False,

    original_dir=gsr_original_dir,
    adv_dir=gsr_adv_dir,
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ARTWORK ATTACK SUMMARY")
print("=" * 70)


print("\nPGD")
print("-" * 40)

print(
    f"Images processed       : {pgd_count}"
)

print(
    f"Successful attacks     : {pgd_success}"
)

if pgd_count > 0:

    pgd_asr = (
        pgd_success
        / pgd_count
        * 100
    )

    print(
        f"Attack success rate    : "
        f"{pgd_asr:.2f}%"
    )

print(
    f"Average generation time: "
    f"{pgd_time:.2f} ms/image"
)


print("\nGSR-PGD")
print("-" * 40)

print(
    f"Images processed       : {gsr_count}"
)

print(
    f"Successful attacks     : {gsr_success}"
)

if gsr_count > 0:

    gsr_asr = (
        gsr_success
        / gsr_count
        * 100
    )

    print(
        f"Attack success rate    : "
        f"{gsr_asr:.2f}%"
    )

print(
    f"Average generation time: "
    f"{gsr_time:.2f} ms/image"
)


print("\n" + "=" * 70)
print("COMPLETE")
print("=" * 70)

# ============================================================
# INSTANTSTYLE GENERATION
# ============================================================

print("\n" + "=" * 70)
print("STARTING GENERATIVE STYLE-MIMICRY EVALUATION")
print("=" * 70)

instantstyle_results = run_instantstyle(
    results_path=results_path
)

print("\nInstantStyle summary:")
print(
    f"Source artworks : "
    f"{instantstyle_results['artworks']}"
)

print(
    f"Generated images: "
    f"{instantstyle_results['generated']}"
)

print(
    f"Output folder   : "
    f"{instantstyle_results['output_dir']}"
)