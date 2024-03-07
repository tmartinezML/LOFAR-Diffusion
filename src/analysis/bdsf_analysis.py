from utils.data_utils import EvaluationDataset
import utils.paths as paths
from utils.paths import LOFAR_SUBSETS
import argparse
import multiprocessing as mp
import multiprocessing.pool
from concurrent.futures import ProcessPoolExecutor as PPEx
from functools import partial
import time
import signal

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
from tqdm.contrib.concurrent import process_map
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


def append_to_csv(df, fpath):
    df.to_csv(fpath, index=False, mode='a', header=not fpath.exists())


def writer(queue, fpaths, write_fns):
    log_file = fpaths[0].parent / 'writer_log.txt'
    i = 0
    def signal_handler(sig, frame):
        print('Keyboard interrupt ignored by writer.')
    signal.signal(signal.SIGINT, signal_handler)
    while True:
        items = queue.get()
        if items == -1:
            print("Closing writer.")
            break
        for item, fpath, write_fn in zip(items, fpaths, write_fns):
            if item is None:
                continue
            write_fn(item, fpath)

        # Write log
        img_id = items[0]['Image_id']
        with open(log_file, 'a' if i else 'w') as f:
            f.write(f'{i},{img_id}\n')
        i += 1


def bdsf_worker_func(img, image_id, q):
    def signal_handler(sig, frame):
        pid = os.getpid()
        print(f'Keyboard interrupt handeled by worker process {pid}.')
        return

    signal.signal(signal.SIGINT, signal_handler)
    # Run bdsf on the image
    bdsf_img = bdsf_on_image(img)

    # bdsf dict:
    # Retrieve the desired attributes of the bdsf image object as dict
    attr_dict = dict_from_bdsf(bdsf_img)
    bdsf_dict = {
        k: attr_dict[k] for k in [
            'model_gaus_arr', 'total_flux_gaus',
        ]
    }
    # Add image_id to the dict
    bdsf_dict['Image_id'] = image_id

    # Catalogs:
    # Retrieve the catalogs of the bdsf image object
    cats = catalogs_from_bdsf(bdsf_img)  # cat_g, cat_s
    for cat in cats:
        if cat is not None:
            cat['Image_id'] = image_id

    # Append to queue
    q.put([bdsf_dict, *cats])


def iterable_func_caller(func, args):
    return func(*args)


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

    # Look for images already processed
    else:
        print('Looking for images already processed...')
        # Open pickle file
        if dicts_path.exists():
            with open(dicts_path, 'rb') as f:
                dicts = pickle.load(f)
            ids = np.array([d['Image_id'] for d in dicts])
            print(f'Removing {len(ids)} processed images from the list.')
            mask = np.in1d(names, ids, invert=True)
            imgs = imgs[mask]
            names = names[mask]
        else:
            print('No images processed yet.')

    # Set up multiprocessing
    manager = mp.Manager()
    q = manager.Queue()
    writer_pool = mp.Pool(1)
    w_res = writer_pool.apply_async(
        writer,
        (
            q,
            [dicts_path, gaul_path, srl_path],
            [append_to_pickle, append_to_csv, append_to_csv]
        )
    )

    image_ids = names if names is not None else range(len(imgs))

    # Run bdsf on all images
    with PPEx(max_workers=10) as worker_pool:

        def signal_handler(sig, frame):
            print('Keyboard interrupt. Closing pools.')
            worker_pool.shutdown(cancel_futures=True)
            q.put(-1)
            writer_pool.close()
            writer_pool.join()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        
        res = list(tqdm(
            worker_pool.map(
                partial(iterable_func_caller, bdsf_worker_func),
                zip(imgs, image_ids, [q] * len(imgs)),
                chunksize=10,
            ),
            total=len(imgs),
        ))
        [print(r) for r in res if r is not None]

            

    # Close the writer pool
    q.put(-1)
    writer_pool.close()
    writer_pool.join()
    s = w_res.successful()
    print(f'Writer success: {s}')
    if not s:
        print(w_res.get())

    print('Done.')


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

    # For testing: deterministic subset (use None for all images)
    n = None

    # Run bdsf on the images
    bdsf_out = bdsf_run(
        dataset.data[:n],
        out_folder=dataset.path.stem,
        names=dataset.names[:n],
        override=arguments.override,
    )