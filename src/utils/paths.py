from pathlib import Path
import os

# Base directories for code base & storage
BASE_PARENT = Path(os.getcwd()).parent
STORAGE_PARENT = Path("/hs/fs08/data/group-brueggen/tmartinez")

# Three main storage folders.
MODEL_PARENT = STORAGE_PARENT / 'model_results'
ANALYSIS_PARENT = STORAGE_PARENT / 'analysis_results'
IMG_DATA_PARENT = STORAGE_PARENT / 'image_data'
# Create symlinks to the code base
for p in [MODEL_PARENT, ANALYSIS_PARENT, IMG_DATA_PARENT]:
    symlink = BASE_PARENT / p.name
    if not symlink.exists():
        symlink.symlink_to(p)
    else:
        assert symlink.resolve() == p, (
            f"Broken folder structure: Symlink {symlink} points to {symlink.resolve()}."
        )

# Folders for different kinds of image data
GEN_DATA_PARENT = IMG_DATA_PARENT / 'generated'
LOFAR_DATA_PARENT = IMG_DATA_PARENT / 'LOFAR'
FIRST_DATA_PARENT = IMG_DATA_PARENT / 'FIRST'

# Other useful folders
PLAYGORUND_DIR = ANALYSIS_PARENT / 'playground'
DEBUG_DIR = MODEL_PARENT / 'debug'

# Train data subsets
LOFAR_SUBSETS = {k: LOFAR_DATA_PARENT / v for k, v in {
    '0-clip_unscaled': 'lofar_120asLimit_80p_0-clipped_f-thr=0_SNR>=5_subset.hdf5',
    '120asLimit_SNR>=5': 'lofar_120asLimit_80p_unclipped_f-thr=0_SNR>=5_subset.hdf5',
    '50asLimit_SNR>=5': 'lofar_50asLimit_80p_unclipped_subset_f-thr=0_SNR>=5.hdf5',
    '1.5LAS_f-thr=0.75': 'lofar_1p5las_80p_unclipped_subset_f-thr=0p75.hdf5',
}.items()}


def cast_to_path(p):
    if isinstance(p, str):
        return Path(p)
    return p


if __name__ == "__main__":

    print("Base directories for code base & storage")
    print(f"\tBASE_PARENT: {BASE_PARENT}")
    print(f"\tSTORAGE_PARENT: {STORAGE_PARENT}")

    print("\nThree main storage folders.")
    print(f"\tMODEL_PARENT: {MODEL_PARENT}")
    print(f"\tANALYSIS_PARENT: {ANALYSIS_PARENT}")
    print(f"\tIMG_DATA_PARENT: {IMG_DATA_PARENT}")

    print("\nFolders for different kinds of image data")
    print(f"\tGEN_DATA_PARENT: {GEN_DATA_PARENT}")
    print(f"\tLOFAR_DATA_PARENT: {LOFAR_DATA_PARENT}")
    print(f"\tFIRST_DATA_PARENT: {FIRST_DATA_PARENT}")

    print("\nOther useful folders")
    print(f"\tPLAYGORUND_DIR: {PLAYGORUND_DIR}")
    print(f"\tDEBUG_DIR: {DEBUG_DIR}")

    print("\nTrain data subsets")
    for k, v in LOFAR_SUBSETS.items():
        print(f"\t{k}: {v}")
