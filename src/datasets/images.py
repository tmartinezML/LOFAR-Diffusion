import tempfile

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from astropy.io import fits
from skimage.transform import resize
from astropy.stats import sigma_clipped_stats

from utils.paths import cast_to_Path


def clip_image(image, f_clip=1.5):
    """
    Clip the image to f_clip*stddev.
    """
    if f_clip is None:
        return np.array(image)

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


def minmax_scaling(image):
    """
    Normalize the image to 0-1.
    """
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    # image = 255 * image
    # image = image.astype('uint8')
    # image = Image.fromarray(image, 'L')
    return image





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
