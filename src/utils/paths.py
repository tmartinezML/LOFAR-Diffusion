from pathlib import Path
import os
from indexed import IndexedOrderedDict

# Base directories for code base & storage
BASE_PARENT = Path(__file__).parent.parent.parent
STORAGE_PARENT = Path("/hs/fs08/data/group-brueggen/tmartinez")

# Three main storage folders.
MODEL_PARENT = STORAGE_PARENT / "model_results"
ANALYSIS_PARENT = STORAGE_PARENT / "analysis_results"
IMG_DATA_PARENT = STORAGE_PARENT / "image_data"
# Create symlinks to the code base
for p in [MODEL_PARENT, ANALYSIS_PARENT, IMG_DATA_PARENT]:
    symlink = BASE_PARENT / p.name
    if not symlink.exists():
        symlink.symlink_to(p)
    else:
        assert (
            symlink.resolve() == p
        ), f"Broken folder structure: Symlink {symlink} points to {symlink.resolve()}."

# Folders for different kinds of image data
GEN_DATA_PARENT = IMG_DATA_PARENT / "generated"
LOFAR_DATA_PARENT = IMG_DATA_PARENT / "LOFAR"
FIRST_DATA_PARENT = IMG_DATA_PARENT / "FIRST"

# Other useful folders
PLAYGROUND_DIR = ANALYSIS_PARENT / "playground"
DEBUG_DIR = MODEL_PARENT / "debug"

# Train data subsets
LOFAR_SUBSETS = IndexedOrderedDict(
    {
        k: (LOFAR_DATA_PARENT / "subsets") / v
        for k, v in {
            "0-clip": "0-clip.hdf5",
            "1.5-clip": "1p5sigma-clip.hdf5",
            "2-clip": "2sigma-clip.hdf5",
            "unclipped": "unclipped.hdf5",
            "0-clip_unscaled": "lofar_120asLimit_80p_0-clipped_f-thr=0_SNR>=5_subset.hdf5",
            "120asLimit_SNR>=5": "lofar_120asLimit_80p_unclipped_f-thr=0_SNR>=5_subset.hdf5",
        }.items()
    }
)

# Paths for training data processing
MOSAIC_DIR = "/hs/fs05/data/AG_Brueggen/nicolasbp/RadioGalaxyImage/data/mosaics_public"
CUTOUTS_DIR = LOFAR_DATA_PARENT / "cutouts"
LOFAR_RES_CAT = LOFAR_DATA_PARENT / "6-LoTSS_DR2-public-resolved_sources.csv"

# Paths for output
PAPER_PLOT_DIR = ANALYSIS_PARENT / "paper_plots"


def cast_to_Path(path):
    """
    Create a Path object from a string.
    """
    match path:
        case Path():
            return path
        case str():
            return Path(path)
        case _:
            raise TypeError(f"Expected Path or str, got {type(path)}")


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
    print(f"\tPLAYGORUND_DIR: {PLAYGROUND_DIR}")
    print(f"\tDEBUG_DIR: {DEBUG_DIR}")

    print("\nTrain data subsets")
    for k, v in LOFAR_SUBSETS.items():
        print(f"\t{k}: {v}")
