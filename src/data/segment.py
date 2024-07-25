import os
import numpy as np
import skimage.draw as sk_draw
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from skimage.measure import label, regionprops
from scipy.ndimage import (
    binary_fill_holes,
    binary_closing,
    binary_opening,
    binary_dilation,
)
from astropy.stats import sigma_clipped_stats

import data.smallest_circle as sc


def remove_small_islands(mask, min_pixels):
    # Label the islands in the mask
    labels = label(mask)

    # Create a new mask of the same shape as the original mask, filled with False
    new_mask = np.zeros(mask.shape, dtype=bool)

    # For each region (island) in the labeled mask
    for region in regionprops(labels):
        # If the region has at least min_pixels
        if region.area >= min_pixels:
            # Set the pixels in the new mask that correspond to this region to True
            new_mask[labels == region.label] = True

    return new_mask


def smooth_mask(mask, closing_kernel=3, opening_kernel=1, min_px=5):
    mask = binary_fill_holes(mask)
    mask = binary_closing(mask, structure=np.ones((closing_kernel,) * 2))
    mask = binary_opening(mask, structure=np.ones((opening_kernel,) * 2))
    mask = remove_small_islands(mask, min_px)
    return mask


def smooth_masks_parallel(masks, max_workers=None, **kwargs):
    results = [None] * len(masks)  # Pre-allocate results list

    # If max_workers is not specified, default to the number of CPUs available
    if max_workers is None:
        max_workers = os.cpu_count()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map of future to mask index for result placement
        future_to_index = {
            executor.submit(smooth_mask, mask, **kwargs): i
            for i, mask in enumerate(masks)
        }

        # Initialize tqdm progress bar
        progress = tqdm(
            as_completed(future_to_index), total=len(masks), desc="Smoothing Masks"
        )

        # Collect results as they complete
        for future in progress:
            index = future_to_index[future]
            try:
                result = future.result()  # Get the result
                results[index] = result  # Place result in correct order
            except Exception as exc:
                print(f"Mask processing generated an exception: {exc}")

    return np.array(results)


def refine_mask(img, mask, n=20, step=0.1, sigma_threshold=5):
    # Initialize refined mask, will be filled with refined regions
    mask_refined = np.zeros_like(mask, dtype=int)

    # Dilated mask to safely exclude signal from background
    mask_dilated = binary_dilation(mask, iterations=3)

    # Go through islands in the mask
    labels = label(mask)
    for region in regionprops(labels):
        region_mask = labels == region.label

        # Elliptic mask for local background. Start with a small ellipse,
        # and increase its size until the encompassed background covers at least
        # n pixels or the whole background region
        c = 1
        ell_mask = elliptic_mask(region, scaling=c)
        while np.sum(ell_mask * (1 - mask_dilated)) < min(
            n * region.num_pixels, (1 - mask_dilated).sum()
        ):
            c += step
            ell_mask = elliptic_mask(region, scaling=c)

        # Local background mask is intersection of the elliptic mask
        # and the background mask
        local_bg_mask = ell_mask * (1 - mask_dilated)

        # Local background stats
        _, med, std = sigma_clipped_stats(img[local_bg_mask == 1], sigma=3, maxiters=5)

        # Local sigma mask
        local_sigma_mask = ((img > med + sigma_threshold * std) * ell_mask).astype(int)

        # Expand region mask to include overlapping islands in the sigma mask
        mask_refined += expand_islands(region_mask, local_sigma_mask)

    mask_refined = (mask_refined > 0).astype(int)
    return mask_refined


def refine_mask_iterative(img, mask, max_iter=10, **kwargs):
    it_levels = np.zeros_like(mask).astype(int)
    for _ in range(max_iter):
        ref_mask = refine_mask(img, mask, **kwargs)
        it_levels += ref_mask
        if np.array_equal(ref_mask, mask):
            break
        mask = ref_mask

    return mask, it_levels


def refine_masks_iterative_parallel(imgs, masks, max_workers=None, **kwargs):
    # Function to be executed in parallel
    def process_task(img, mask):
        return refine_mask_iterative(img, mask, **kwargs)[0]

    if max_workers is None:
        max_workers = os.cpu_count()

    # List to hold the results
    results = [None] * len(imgs)

    # Use ThreadPoolExecutor to parallelize the task
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and create a mapping of future to its index
        future_to_index = {
            executor.submit(process_task, img, mask): i
            for i, (img, mask) in enumerate(zip(imgs, masks))
        }

        # Initialize tqdm progress bar
        progress = tqdm(
            as_completed(future_to_index), total=len(imgs), desc="Refining Masks:"
        )

        # As tasks complete, update the results list and the progress bar
        for future in progress:
            index = future_to_index[future]
            try:
                result = future.result()
                results[index] = result
            except Exception as exc:
                print(f"Image {index} generated an exception: {exc}")

    return np.array(results)


def expand_islands(mask1, mask2):
    # Label the islands in the masks
    labels1 = label(mask1)
    labels2 = label(mask2)

    # Copy 1st mask
    new_mask = np.copy(mask1)

    # For each region (island) in the labeled first mask
    for region in regionprops(labels1):

        # Define region mask
        region_mask = labels1 == region.label

        # Find labels of all islands in 2nd mask
        # that overlap with the current region
        overlapping_labels = np.unique(labels2[np.nonzero(labels2 * region_mask)])

        if len(overlapping_labels) > 0:
            # Create a mask of the overlapping islands
            overlap_mask = np.isin(labels2, overlapping_labels)
            new_mask += overlap_mask

    return new_mask


def get_regions(mask):
    labels = label(mask)
    return [region for region in regionprops(labels)]


def get_circle(island_mask):
    signal_points = np.flip(np.argwhere(island_mask), axis=1)  # Array of [x, y] points

    x, y, r = sc.make_circle(signal_points)
    circle = np.round(np.array([x, y, np.ceil(r)])).astype(int)


def elliptic_mask(region, scaling=1, shape=(200,) * 2):

    ell_mask = np.zeros(shape)
    rr, cc = sk_draw.ellipse(
        region.centroid[0],
        region.centroid[1],
        max(region.major_axis_length, 2) / 2 * scaling,
        max(region.minor_axis_length, 2) / 2 * scaling,
        rotation=region.orientation,
        shape=shape,
    )
    ell_mask[rr, cc] = 1
    return ell_mask
