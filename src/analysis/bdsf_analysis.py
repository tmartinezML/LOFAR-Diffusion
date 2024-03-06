from utils.data_utils import EvaluationDataset
import utils.paths as paths
from utils.paths import LOFAR_SUBSETS
import argparse

import sys
import os
import pickle
import dill
import tempfile
from collections.abc import Iterable
from numbers import Number
from pathlib import Path

from astropy.io import fits
import bdsf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

import logging
import functools


def disable_logging(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logging.disable(logging.CRITICAL + 1)
        result = func(*args, **kwargs)
        logging.disable(logging.NOTSET)
        return result
    return wrapper


# decorater used to block function printing to the console
def block_printing(func):
    def func_wrapper(*args, **kwargs):
        # block all printing to the console
        sys.stdout = open(os.devnull, 'w')
        # call the method in question
        value = func(*args, **kwargs)
        # enable all printing to the console
        sys.stdout = sys.__stdout__
        # pass the return value of the method back
        return value

    return func_wrapper


@disable_logging
@block_printing
def bdsf_on_image(img: np.ndarray):
    '''
    Run bdsf on a single image.
    '''
    img = img.squeeze()

    # Set up the header, which contains information required for bdsf
    beam_size = 0.001666  # 6 arcsec
    # Angular size of the images is 50 arcsec, pixel size is 80x80.
    # Pixel size in deg:
    px_size_deg = 50 / 3600 / 80
    header_dict = {
        "CDELT1": -px_size_deg,  # Pixel size in deg (1,5 arcsec)
        "CUNIT1": "deg",
        "CTYPE1": "RA---SIN",
        "CDELT2": px_size_deg,
        "CUNIT2": "deg",
        "CTYPE2": "DEC--SIN",
        "CRVAL4": 143650000.0,  # Frequency in Hz
        "CUNIT4": "HZ",
        "CTYPE4": "FREQ",
    }

    # Set up hdu with header and image data for temporary fits file,
    # because bdsf only accepts fits files
    hdu = fits.PrimaryHDU(data=img, header=fits.Header(header_dict))

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.fits') as f:

        # Write the hdu to tmp fits file
        fits.HDUList([hdu]).writeto(f.name, overwrite=True)

        # Run bdsf on the fits file
        img = bdsf.process_image(
            f.name,
            thresh_isl=1.5,
            thresh_pix=0.5,
            beam=(beam_size, beam_size, 0),
            mean_map='const',
            rms_map=False,
            thresh='hard',
            quiet=True,
        )

        # Remove log file
        os.remove(f'{f.name}.pybdsf.log')

    # Return bdsf image object
    return img

@block_printing
def catalogs_from_bdsf(img: bdsf.image.Image):
    cat_list = []

    # Loop through both cat types
    for cat_type in ['gaul', 'srl']:

        # Create a temporary csv file
        with tempfile.NamedTemporaryFile(suffix='.csv') as f:

            # Write the catalog to the csv file
            img.write_catalog(
                outfile=f.name,
                clobber=True,
                catalog_type=cat_type,
                format='csv',
            )

            # Read the csv file
            try:
                catalog = pd.read_csv(f.name, skiprows=5,
                                      skipinitialspace=True)

            except pd.errors.EmptyDataError:
                catalog = None

            # Append to list
            cat_list.append(catalog)

    return cat_list


def dict_from_bdsf(img: bdsf.image.Image):
    # Get all attributes of the bdsf image object
    # that are numbers or numpy arrays
    d = {
        k: v for k, v in img.__dict__.items()
        if isinstance(v, Number) or isinstance(v, np.ndarray)
    }
    return d


def append_to_pickle(obj, fpath):
    if fpath.exists():
        with open(fpath, 'rb') as f:
            data = pickle.load(f)
    else:
        data = []
    data.append(obj)
    with open(fpath, 'wb') as f:
        pickle.dump(data, f)


def bdsf_run(
        imgs: Iterable,
        out_folder: str | Path,
        out_parent=paths.ANALYSIS_PARENT,
        names: Iterable[str] = None,
        override=False,
):
    '''
    Run bdsf on a set of images and save the resulting list as pickle file.
    '''
    # Set up the output folder
    out_path = out_parent / out_folder
    out_path.mkdir(exist_ok=True)

    # Set up the output file paths
    dicts_path = out_path / f'{out_folder}_bdsf_dicts.pkl'
    gaul_path = out_path / f'{out_folder}_bdsf_gaul.csv'
    srl_path = out_path / f'{out_folder}_bdsf_srl.csv'

    # If override is True, delete the files if they exist
    if override:
        for fpath in [dicts_path, gaul_path, srl_path]:
            if fpath.exists():
                fpath.unlink()

    # Counter for skipped sources.
    n_skip = 0

    # Run bdsf on all images
    for i, img in enumerate(tqdm(imgs)):

        # Set the image_id
        image_id = names[i] if names is not None else i

        # If files exist, check if image has already been processed.
        # Check only for gaussians, because they are necessary for the srl
        # catalog anyway.
        if gaul_path.exists():
            if image_id in pd.read_csv(gaul_path).Image_id.values:
                n_skip += 1
                print(
                    f"Skipping image {image_id}, already processed."
                    f" ({n_skip} skipped so far.)"
                )
                continue

        # Run bdsf on the image
        bdsf_img = bdsf_on_image(img)

        # bdsf dict:
        # Retrieve the desired attributes of the bdsf image object as dict
        bdsf_dict = dict_from_bdsf(bdsf_img)
        # Add image_id to the dict
        bdsf_dict['Image_id'] = image_id
        # Write to pickle file
        append_to_pickle(bdsf_dict, dicts_path)

        # Catalogs:
        # Retrieve the catalogs of the bdsf image object
        cat_g, cat_s = catalogs_from_bdsf(bdsf_img)
        # Append catalog data to csv files
        for cat, fpath in zip([cat_g, cat_s], [gaul_path, srl_path]):
            # None is returned if no sources are found
            if cat is not None:
                # Add image_id column
                cat['Image_id'] = image_id
                # Append to csv file
                cat.to_csv(
                    fpath,
                    index=False,
                    mode='a',
                    header=not fpath.exists(),
                )

    return dicts_path, gaul_path, srl_path


def bdsf_plot(
        img,
        keys=['ch0_arr', 'resid_gaus_arr', 'model_gaus_arr']
):
    fig = plt.figure(figsize=(10, 15))
    for i, key in enumerate(keys):
        # add subplot
        ax = fig.add_subplot(1, len(keys), i + 1)
        image = getattr(img, key)
        ax.imshow(image.T)
        ax.set_title(key)
        ax.set_axis_off()
    fig.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--override", required=False,
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    arguments = parser.parse_args()

    # Load the dataset
    dataset = EvaluationDataset(
        LOFAR_SUBSETS['unclipped_SNR>=5_50asLimit'])

    # For testing: deterministic subset
    # n = 3

    # Run bdsf on the images
    bdsf_out = bdsf_run(
        dataset.data[:],
        out_folder=dataset.path.stem,
        names=dataset.names[:],
        override=arguments.override,
    )
