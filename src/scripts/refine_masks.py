import h5py
import numpy as np

import utils.paths as paths
import data.segment as seg
from data.datasets import EvaluationDataset
from data.transforms import minmax_scale_batch

# Load the dataset
dset_file = paths.LOFAR_SUBSETS["200p"]
dset = EvaluationDataset(dset_file, img_size=200)

# For testing:
# idxs = np.random.choice(len(dset), 100, replace=False)
# dset.index_slice(idxs)

# Get images and masks
imgs = minmax_scale_batch(dset.data).numpy()
masks = dset.island_labels

# Smooth and refine
masks = seg.smooth_masks_parallel(masks)
ref_masks = seg.refine_masks_iterative_parallel(imgs, masks)

# Save to dataset
dset_name = "masks_refined"
with h5py.File(dset_file, "a") as f:
    if dset_name in f:
        del f[dset_name]
    f.create_dataset(dset_name, data=ref_masks)
