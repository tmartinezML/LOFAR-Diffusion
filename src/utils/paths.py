from pathlib import Path

BASE_PARENT = Path("/home/bbd0953/diffusion")
# Symlinked to base parent
MODEL_PARENT = Path("/storage/tmartinez/model_results")
DEBUG_DIR = MODEL_PARENT / 'debug'
IMG_DATA_PARENT = BASE_PARENT / 'image_data'
GEN_DATA_PARENT = IMG_DATA_PARENT / 'generated'
LOFAR_DATA_PARENT = IMG_DATA_PARENT / 'LOFAR'
FIRST_DATA_PARENT = IMG_DATA_PARENT / 'FIRST'
ANALYSIS_PARENT = BASE_PARENT / 'analysis_results'
PLAYGORUND_DIR = ANALYSIS_PARENT / 'playground'

# Train data subsets
LOFAR_SUBSETS = {k: LOFAR_DATA_PARENT / v for k, v in {

    'zero-clipped_PNG': 'lofar_zoom_unclipped_subset_80p',
    'unclipped_H5': 'lofar_1p5las_80p_unclipped_subset_f-thr=0p75.hdf5',
    'unclipped_abs_dset-norm': 'lofar_zoom_unclipped_subset_80p_abs_dset-norm.hdf5',
    'unclipped_abs_subset-norm': 'lofar_zoom_unclipped_subset_80p_abs_subset-norm.hdf5',
    'unclipped_SNR>=5': 'lofar_1p5las_unclipped_f-thr=0_SNR>=5_subset_80p.hdf5',
    'unclipped_SNR>=5_50asLimit': 'lofar_50asLimit_unclipped_f-thr=0_SNR>=5_subset_80p.hdf5',
    '1sigma-clipped': 'lofar_1p5las_1sigma-clipped_subset_80p.hdf5'

}.items()}
