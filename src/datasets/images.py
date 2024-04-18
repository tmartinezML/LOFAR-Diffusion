import tempfile
import numbers

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from astropy.io import fits
from skimage.transform import resize
from astropy.stats import sigma_clipped_stats

from datasets.cutouts import save_images_hpy5
from utils.paths import cast_to_Path


from pathlib import Path
from collections.abc import Sequence
import h5py
import pandas as pd
import numpy as np
import numpy.ma as ma
from tqdm import tqdm

from astropy.stats import sigma_clip, sigma_clipped_stats


def clip_image(image, f_clip=1.5):
    """
    Clip the image to f_clip*stddev.
    """
    if f_clip is None:
        return np.array(image)

    elif f_clip == 0:
        return np.clip(image, 0, np.inf)

    _, _, stddev = sigma_clipped_stats(data=image, sigma=3.0, maxiters=10)
    image = np.clip(image, f_clip * stddev, np.inf)
    return image


def resize_image(image, size=128):
    """
    Resize the image to size*size.
    """
    if size == image.shape[-1]:
        return image
    image = resize(image, (size, size))
    return image


def clip_cutouts(
    cutout_file: str | Path,
    f_clip: numbers.Number | list | tuple = 0,
):
    # Load cutouts and catalog from file
    print(f"Reading file {cutout_file.name}...")
    cutout_file = cast_to_Path(cutout_file)
    with h5py.File(cutout_file, "r") as f:
        images = np.array(f["cutouts"])
    catalog = pd.read_hdf(cutout_file, key="catalog")
    catalog_flag = ~catalog["Problem_cutout"]

    assert len(images) == sum(
        catalog_flag
    ), f"Number of images {len(images)} and catalog entries {sum(catalog_flag)} do not match."
    
    # Clip images
    if isinstance(f_clip, numbers.Number):
        f_clip = [f_clip]

    for clip in f_clip:
        images_clipped = []
        clip_flag = np.zeros(len(catalog[catalog_flag]), dtype=bool)
        sigma_snr_vals = -1 * np.ones(len(catalog[catalog_flag]), dtype=bool)

        # Loop through images for clipping and sigma-snr calculation
        for i, image in enumerate(tqdm(images, desc=f"Clipping {clip}-sigma...")):
            # Clipping
            try:
                image = clip_image(image, f_clip=clip)
                clip_flag[i] = True
                images_clipped.append(image)
            except Exception:
                pass

            # Sigma SNR:
            try:
                sigma_snr_vals[i] = sigma_snr(image, threshold=5)
            except Exception:
                pass

        # Write values into catalog
        print("Updating catalog...")
        catalog[f"Clipped_{clip}sigma"] = np.zeros(len(catalog), dtype=bool)
        catalog[f"Sigma_SNR_{clip}sigma"] = -1 * np.ones(len(catalog), dtype=bool)

        catalog.loc[catalog_flag, f"Clipped_{clip}sigma"] = clip_flag
        catalog.loc[catalog_flag, f"Sigma_SNR_{clip}sigma"] = sigma_snr_vals

        # Save clipped images to same file as new dataset
        print("Saving images...")
        save_images_hpy5(
            images_clipped,
            cutout_file,
            dset_name=f"cutouts_{clip}sigma",
        )

    # Replace catalog in file
    print("Saving catalog...")
    catalog.to_hdf(
        cutout_file,
        key="catalog",
        mode="a",
    )
    print("Done.")


def process_cutouts(
    catalog: pd.DataFrame,
    cutout_dir,
    f_clip=1.5,
    size=80,
    comment=None,
    norm_dataset=False,
    absolute=False,
):
    """
    Process the cutouts by clipping, resizing and normalizing.
    """
    # Set paths
    cutout_dir = cast_to_Path(cutout_dir)
    export_base = cutout_dir.parent

    # Set out file name
    out_file_name = f"{cutout_dir.name}_processed"
    # Clipping value:
    if f_clip is not None:
        out_file_name += f"_{f_clip}".replace(".", "p") + "sigma-clipped"
    else:
        out_file_name += "_unclipped"
    # Size of images:
    out_file_name += f"_{size}p"
    if comment is not None:
        out_file_name += f"_{comment}"
    out_file = export_base / f"{out_file_name}.hdf5"

    # Get source files
    source_files = list(cutout_dir.iterdir())

    # Create temporary files
    print("Creating temporary files...")
    tmp_clipped, tmp_resized, tmp_norm = [
        tempfile.NamedTemporaryFile(
            delete=True, dir=export_base, prefix="tmp_process_cutouts_"
        )
        for _ in range(3)
    ]
    shape = (len(source_files), size, size)
    imgs_resized, imgs_norm = [
        np.memmap(tmp.name, dtype=np.float32, mode="w+", shape=shape)
        for tmp in [tmp_resized, tmp_norm]
    ]
    dummy_arr = np.full(shape, -1)
    for a in [imgs_resized, imgs_norm]:
        a[:] = dummy_arr

    names_clipped, names_resized, names_norm = [], [], []

    for i, source_file in tqdm(
        enumerate(source_files), desc="Processing cutouts...", total=len(source_files)
    ):
        source_name = source_file.name.replace(".fits", "")
        try:
            # Open fits file
            with fits.open(source_file, memmap=False) as hdul:
                image = hdul[0].data
                image = image.view(np.recarray)

            # Clip
            image = clip_image(image, f_clip=f_clip)
            with open(tmp_clipped.name, "ab") as tmp:
                np.save(tmp, image)
            names_clipped.append(source_name)

            # Resize
            image = resize_image(image, size=size)
            imgs_resized[i] = image
            names_resized.append(source_name)

            # Normalize
            if not norm_dataset:
                image = minmax_scaling(image)
                imgs_norm[i] = image
                names_norm.append(source_name)

        except Exception as exc:
            print(f"Problem with {source_file.name}. \n {exc}")
            source_loc = catalog["Source_Name"] == source_file.name.replace(".fits", "")
            catalog.loc[source_loc, "Problem_preprocess"] = True

    # Read clipped images from temporary file
    print("Reading clipped images from temporary file...")
    with open(tmp_clipped.name, "rb") as tmp:
        imgs_clipped = []
        while True:
            try:
                imgs_clipped.append(np.load(tmp).astype(np.float32))
            except EOFError:
                break

    # Remove elements that are still dummy arrays
    print("Removing dummy arrays...")
    for a in [imgs_resized, imgs_norm]:
        a = a[a != dummy_arr]

    # Normalize over whole dataset
    if norm_dataset:
        print("Normalizing over whole dataset...")
        imgs_resized = np.array(imgs_resized)
        resized_copy = imgs_resized.copy()
        if absolute:
            resized_copy = np.abs(resized_copy)
        norm_min, norm_max = np.min(resized_copy), np.max(resized_copy)
        imgs_norm = minmax_scaling(resized_copy)
        names_norm = names_resized

    # Save images
    if out_file.exists():
        out_file.unlink()

    print("Saving images...")
    save_images_hpy5(
        imgs_clipped,
        out_file,
        dset_name="images_clipped",
        src_names=np.array(names_clipped),
        attributes={"f_clip": f_clip if f_clip is not None else "None"},
        dtype=h5py.vlen_dtype(imgs_clipped[0].dtype),
    )
    save_images_hpy5(
        np.array(imgs_resized),
        out_file,
        dset_name="images_resized",
        src_names=np.array(names_resized),
        attributes={"size": size},
    )
    save_images_hpy5(
        np.array(imgs_norm),
        out_file,
        dset_name="images_norm",
        src_names=np.array(names_norm),
        attributes={
            "norm_dataset": norm_dataset,
            "absolute": absolute,
            "norm_min": norm_min if norm_dataset else "Per-Img",
            "norm_max": norm_max if norm_dataset else "Per-Img",
        },
    )

    # Remove temporary files
    [tmp.close() for tmp in [tmp_clipped, tmp_resized, tmp_norm]]

    return catalog, out_file


def remove_problematic_sources(catalog):
    """
    Remove the sources that have a morphology class of -1, or that have NaNs in the cutouts.
    """
    catalog = catalog[
        (catalog["Morphology"] != -1)
        & (catalog["Nans_cutout"] == 0)
        & (catalog["Problem_cutout"] == 0)
    ]

    return catalog


def sigma_mask(img, threshold=5):
    _, med, std = sigma_clipped_stats(img)
    return img > med + threshold * std


def sigma_snr(img, threshold=5):
    mask = sigma_mask(img, threshold)

    if mask.sum() == 0:
        return sigma_snr(img, threshold - 0.5)

    if img[~mask].sum() == 0:
        print("No pixels below threshold.")
        return -1

    return img[mask].sum() / img[~mask].sum() * (~mask).sum() / mask.sum()


def sigma_snr_set(images, threshold=5):
    return np.array([sigma_snr(img, threshold) for img in tqdm(images, desc="\t")])


def check_nans(images, names):
    print("Checking for nans...")
    for i, img in tqdm(enumerate(images), desc="\t", total=len(images)):
        if np.isnan(img).any():
            print(f"\t{names[i]} has nans.")


def edge_pixels(img):
    return np.concatenate([img[0], img[-1], img[1:-1, 0], img[1:-1, -1]])


def edge_threshold_mask(images, threshold=0.8):
    edge_pixel_arr = np.array([edge_pixels(img) for img in tqdm(images, desc="\t")])
    edge_mask = edge_pixel_arr.max(axis=1) > threshold
    return edge_mask


def broken_image_mask(images):
    # This will NOT work on clipped images!!
    images_collapsed = images.reshape(images.shape[0], -1)
    minvals = images_collapsed.min(axis=1)
    return (images_collapsed == np.expand_dims(minvals, 1)).sum(axis=1) > 1


def threshold_mask(
    catalog: pd.DataFrame,
    label: str,
    threshold: int | float | Sequence,
):
    """
    Create a mask for a catalog based on a threshold value. Returns a mask
    where True indicates that the source should be removed.

    Parameters
    ----------
    catalog : pd.DataFrame
        The catalog to use.
    label : str
        The column label to use.
    threshold : int | float | Sequence
        The threshold value(s) to use. If a sequence is provided, the mask will
        be created using the values in the sequence as thresholds. If a single
        value is provided, the mask will be created using that value as the
        threshold.

    Returns
    -------
    mask : pd.Series
        The mask for the catalog.
    """
    match threshold:
        case Sequence():
            mask = catalog[label].between(threshold[0], threshold[1])

        case int() | float() if (threshold > 0):
            mask = catalog[label] > threshold

        case int() | float() if (threshold == 0.0):
            print(f"\tZero {label} threshold, no cuts applied.")
            return pd.Series(False, index=catalog.index)

        case _:
            raise ValueError(f"Invalid {label} threshold value: {threshold}")

    print(f"\t{(~mask).sum():_} sources below {label} threshold.\n")
    return ~mask


def create_subset(
    cutouts_file,
    subset_name,
    clip_sigma=None,
    las_thr=0.0,
    flux_thr=0.0,
    peak_flux=False,
    SNR_thr=None,
    edge_thr=0.8,
):
    # Read cutouts file
    img_container = h5py.File(cutouts_file, "r")

    # Specify dataset tag
    tag = f"cutouts_{clip_sigma}sigma" if clip_sigma is not None else "cutouts"

    # See if dataset is available. If not, clip images first to create.
    if tag not in img_container:
        # Close img container
        img_container.close()
        print(f"\t{tag} dataset not found. Clipping images...")
        clip_cutouts(cutouts_file, f_clip=clip_sigma)
        img_container = h5py.File(cutouts_file, "r")

    # Read images
    print(f"Reading images: {tag}...")
    images = np.array(f[tag])
    print(f"\t{len(images):_} images in catalog.\n")

    # Get the catalog
    print(f"Reading catalog...")
    catalog = pd.from_hdf(cutouts_file, key="catalog")
    if clip_sigma is not None:
        catalog = catalog[catalog[f"Clipped_{clip_sigma}sigma"]]
    print(f"\t{len(catalog):_} sources in catalog.\n")
    n0 = len(catalog)

    print("Creating catalog masks...")

    # Problematic sources
    print("\tProblematic sources:")
    p_cols = [c for c in catalog.columns if "Problem" in c or "Nans" in c]
    p_cat = catalog[p_cols]
    print(*[f"\t{el}\n" for el in p_cat.sum().items()])
    problem_mask = p_cat.sum(axis=1).astype(bool)
    print(f"\t{problem_mask.sum():_} sources with problems.\n")

    # LAS Threshold
    print("\tLAS threshold:")
    las_mask = threshold_mask(catalog, "LAS", las_thr)
    print(f"\t{las_mask.sum():_} sources below threshold.\n")

    # Flux threshold
    print("\tFlux threshold:")
    flux_mask = threshold_mask(
        catalog, "Peak_flux" if peak_flux else "Total_flux", flux_thr
    )
    print(f"\t{flux_mask.sum():_} sources below threshold.\n")

    # SNR threshold
    SNR_mask = np.zeros(len(images), dtype=bool)
    if SNR_thr is not None:
        print("\tSNR threshold:")
        SNR_mask = catalog[f"Sigma_SNR_{clip_sigma}sigma"] < SNR_thr
        print(f"\t{SNR_mask.sum():_} sources below threshold.\n")

    # Edge threshold
    print("\tEdge pixel threshold:")
    edge_mask = edge_threshold_mask(images, threshold=edge_thr)
    print(f"\t{edge_mask.sum():_} sources with edge pixels above threshold.\n")

    # Broken images
    print("\tBroken images:")
    broken_mask = broken_image_mask(images)
    print(f"\t{broken_mask.sum():_} sources with broken image.\n")

    # Combine the masks to create the 'cleaned' catalog mask
    print("\tCombining masks...")
    subset_mask = ~(
        problem_mask | las_mask | flux_mask | SNR_mask | edge_mask | broken_mask
    )
    n_bad = (~subset_mask).sum()
    print(f"\t{n_bad:_} sources will be removed. {n0 - n_bad:_} sources in subset.\n")

    # Filter dataset
    print("Applying mask...")
    images_subset = images[subset_mask]
    catalog_subset = catalog[subset_mask]

    # Create the subset in new file
    print(f"Saving subset to file...")
    subset_file = Path(cutouts_file).parent.parent / f"{subset_name}.hdf5"
    with h5py.File(subset_file, "w") as f:
        img_dataset = f.create_dataset("images", data=images_subset)
        img_dataset.attrs["las_threshold"] = las_thr
        img_dataset.attrs["flux_threshold"] = flux_thr
        img_dataset.attrs["peak_flux"] = peak_flux
        img_dataset.attrs["processed_file"] = str(cutouts_file)
        processed_attrs = {
            key: value
            for dset in img_container.values()
            for (key, value) in dset.attrs.items()
        }
        img_dataset.attrs.update(processed_attrs)

    print("Saving catalog to file...")
    catalog_subset.to_hdf(subset_file, key="catalog", mode="a")

    print(f"Subset file created:\n\t{subset_file}\n")
