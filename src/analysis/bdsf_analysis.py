import os
import sys
import signal
import shutil
import psutil
import pickle
import logging
import argparse
import warnings
import tempfile
import traceback
import functools
import multiprocessing as mp
from pathlib import Path
from numbers import Number
from collections.abc import Iterable
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor as PPEx

import bdsf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from astropy.io import fits

import utils.paths as paths
from utils.data_utils import EvaluationDataset


def disable_logging(func):
    @functools.wraps(func)
    #
    def wrapper(*args, **kwargs):
        logging.disable(logging.CRITICAL + 1)
        result = func(*args, **kwargs)
        logging.disable(logging.NOTSET)
        return result
    return wrapper


# decorater used to block function printing to the console
def redirect_printing(func):
    def func_wrapper(*args, **kwargs):
        # Redirect all printing to the console
        sys.stdout = open(os.devnull, 'w')
        # call the method in question
        try:
            value = func(*args, **kwargs)
        except RuntimeError as e:
            sys.stdout = sys.__stdout__
            raise e
        # enable all printing to the console
        sys.stdout = sys.__stdout__
        # pass the return value of the method back
        return value

    return func_wrapper


# @disable_logging
# @block_printing
def bdsf_on_image(img: np.ndarray, ang_size=120, px_size=80,
                  id=None,
                  tmpdir=paths.ANALYSIS_PARENT / 'tmp',
                  logdir=None,
                  ):
    '''
    Run bdsf on a single image.
    '''
    img = img.squeeze()

    # Add small amount of noise, otherwise sigma-clipping algorithm called
    # within bdsf.process_image (functions.bstat) might not converge
    z = np.random.normal(0, scale=min(img.max(), 1) * 1e-2, size=img.shape)
    img += z

    # Set up the header, which contains information required for bdsf
    beam_size = 0.001667  # 6 arcsec

    # Pixel size in deg:
    px_size_deg = ang_size / 3600 / px_size

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
    with tempfile.NamedTemporaryFile(prefix=id, suffix='.fits', dir=tmpdir) as f:

        # Write the hdu to tmp fits file
        fits.HDUList([hdu]).writeto(f.name, overwrite=True)

        img = bdsf.process_image(
            f.name,
            thresh_isl=1.5,
            thresh_pix=0.5,
            beam=(beam_size, beam_size, 0),
            mean_map='const',
            rms_map=False,
            thresh='hard',
            quiet=True,
            debug=True,
        )

        # (Re)move log file
        if logdir is not None:
            os.rename(
                tmpdir / f'{f.name}.pybdsf.log',
                logdir / f'{id}.pybdsf.log'
            )
        else:
            os.remove(f'{f.name}.pybdsf.log')

    # Return bdsf image object
    return img


@redirect_printing
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


def write_to_pickle(obj, fpath):
    Image_id = obj['Image_id']
    with open(fpath / f'{Image_id}.pkl', 'wb') as f:
        pickle.dump(obj, f)


def append_to_csv(df, fpath):
    df.to_csv(fpath, index=False, mode='a', header=not fpath.exists())


def writer(queue, q_pbar, fpaths, write_fns):
    print('Writer started.')
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

        elif items == 0:
            i += 1
            q_pbar.put(1)
            continue

        else:
            for item, fpath, write_fn in zip(items, fpaths, write_fns):
                if item is None:
                    continue
                write_fn(item, fpath)

            # Write log
            img_id = items[0]['Image_id']
            with open(log_file, 'a' if i else 'w') as f:
                f.write(f'{i},{img_id}\n')

            i += 1
            q_pbar.put(1)


def kill_child_processes(parent_pid, sig=signal.SIGTERM):

    try:
        parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return
    children = parent.children(recursive=True)

    for process in children:
        try:
            process.send_signal(sig)
        except psutil.NoSuchProcess:
            pass


def bdsf_worker_func(img, image_id, q, bdsf_kwargs):

    # Handle keyboard interrupts
    def signal_handler(sig, frame):
        pid = os.getpid()
        print(f'Keyboard interrupt handeled by worker process {pid}.')
        return

    signal.signal(signal.SIGINT, signal_handler)

    # Run bdsf on the image
    bdsf_img = bdsf_on_image(img, id=image_id, **bdsf_kwargs)

    # bdsf dict:
    # Retrieve the desired attributes of the bdsf image object as dict
    attr_dict = dict_from_bdsf(bdsf_img)
    bdsf_dict = {
        k: attr_dict[k] for k in [
            'model_gaus_arr', 'total_flux_gaus', 'ngaus', 'nsrc',
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

    # Append to writer queue
    q.put([bdsf_dict, *cats])


def bdsf_wrapper(*args):
    try:
        bdsf_worker_func(*args)
    except Exception as e:
        args[2].put(0)
        raise e


def iterable_func_caller(func, args):
    return func(*args)


def progress_bar_worker(q_pbar, n_tot):
    print('Progress bar worker started.')

    def signal_handler(sig, frame):
        print('Keyboard interrupt ignored by progress bar worker.')
        q_pbar.put(-1)

    signal.signal(signal.SIGINT, signal_handler)
    with tqdm(total=n_tot, smoothing=0.1) as pbar:
        while True:
            sig = q_pbar.get()
            match sig:
                case 0:
                    print('Test signal received.')
                case -1:
                    print('Closing progress bar.')
                    break
                case 1:
                    pbar.update(1)
                case _:
                    print(f'Unknown signal: {sig}')


def bdsf_run(
        imgs: Iterable,
        out_folder: str | Path,
        out_parent=paths.ANALYSIS_PARENT,
        names: Iterable[str] = None,
        override=False,
        max_workers=96,
        **bdsf_kwargs,
):
    '''
    Run bdsf on a set of images and save the resulting list as pickle file.
    '''
    # Set tmp directory
    tmp_dir = paths.ANALYSIS_PARENT / 'tmp'
    os.environ['TMPDIR'] = str(tmp_dir)
    warnings.filterwarnings('ignore')

    # Set up the output folder
    out_path = out_parent / out_folder / 'bdsf'
    out_path.mkdir(exist_ok=True)

    # Set up the output file paths
    dicts_path = out_path / f'dicts'
    logs_path = out_path / f'logs'
    err_path = out_path / f'errors'
    gaul_path = out_path / f'bdsf_gaul.csv'
    srl_path = out_path / f'bdsf_srl.csv'

    # If override is True, delete the files if they exist
    if override:
        for fpath in [
            dicts_path, logs_path, err_path, gaul_path, srl_path,
        ]:
            if fpath.exists():
                fpath.unlink() if fpath.is_file() else shutil.rmtree(fpath)

    # Look for images already processed
    else:
        print('Looking for images already processed...')

        if dicts_path.exists():
            # Open pickle files in dicts_path
            pkl_files = list(dicts_path.glob('*.pkl'))
            ids = np.array([f.stem for f in pkl_files])
            print(f'Removing {len(ids)} processed images from the list.')
            mask = np.in1d(names, ids, invert=True)
            imgs = imgs[mask]
            names = names[mask]
        else:
            print('No images processed yet.')

    logs_path.mkdir(exist_ok=True)
    err_path.mkdir(exist_ok=True)
    dicts_path.mkdir(exist_ok=True)

    # This will assign an id to every image/job
    print('Getting image ids...')
    image_ids = names if names is not None else range(len(imgs))
    image_ids = [str(i) for i in image_ids]

    # Prepare helper processes: writer and progress bar
    manager = mp.Manager()
    q = manager.Queue()
    q_pbar = manager.Queue()
    helper_pool = mp.Pool(2)

    # Start the writer process
    w_res = helper_pool.apply_async(
        writer,
        (
            q, q_pbar,
            [dicts_path, gaul_path, srl_path],
            [write_to_pickle, append_to_csv, append_to_csv]
        )
    )

    # Start the progress bar process
    pbar_res = helper_pool.apply_async(
        progress_bar_worker,
        (q_pbar, len(imgs))
    )

    # Launch the worker pool
    with PPEx(max_workers=max_workers) as worker_pool:

        # Handle keyboard interrupts
        def signal_handler(sig, frame):
            print('Keyboard interrupt. Closing pools.')
            worker_pool.shutdown(cancel_futures=True, wait=False)
            q.put(-1)
            q_pbar.put(-1)
            helper_pool.close()
            helper_pool.join()
            kill_child_processes(os.getpid())
            for f in tmp_dir.iterdir():
                if f.suffix in ['.log', '.fits']:
                    f.unlink()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        # Add log dir to bdsf kwargs
        bdsf_kwargs['logdir'] = logs_path

        # Run bdsf on the images
        print(f'Running bdsf on images. Parent process: {os.getpid()}.')
        '''
        res = worker_pool.map(
            partial(iterable_func_caller, bdsf_worker_func),
            zip(imgs, image_ids, [q] * len(imgs), [bdsf_kwargs] * len(imgs)),
            chunksize=10,
        )
        '''
        fut = []
        for img, image_id in zip(imgs, image_ids):
            f = worker_pool.submit(
                bdsf_wrapper,
                img, image_id, q, bdsf_kwargs
            )
            fut.append(f)

        # If any exceptions were raised, print to out file
        err_count = 0
        for i, f in enumerate(fut):
            res = f.exception()
            if res is not None:
                with open(err_path / f'{image_ids[i]}.txt', 'w') as f:
                    traceback.print_exception(res, file=f)
                err_count += 1
            else:
                # Check if there is an error file from previous runs,
                # if so remove.
                err_file = err_path / f'{image_ids[i]}.txt'
                if err_file.exists():
                    err_file.unlink()
                    
        print(f'{err_count} errors were encountered - check error directory.')

        # [print(r := f.result()) for f in fut if r is not None]
        # [print(r.successful()) for r in res if r is not None]

    # Finish helper processes
    print('Closing pools.')
    q.put(-1)
    q_pbar.put(-1)
    helper_pool.close()
    helper_pool.join()

    # Check if writer was successful
    s = w_res.successful()
    print(f'Writer success: {s}')
    if not s:
        print(w_res.get())

    # Check if progress bar worker was successful
    s = pbar_res.successful()
    print(f'Progress bar worker success: {s}')
    if not s:
        print(pbar_res.get())

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

    out_folder = paths.ANALYSIS_PARENT / 'Fmax_Context_Dropout/unconditioned'
    data_path = (
        out_folder
        / 'Fmax_Context_Dropout_samples_10000.pt'
    )

    # Load the dataset
    dataset = EvaluationDataset(data_path)

    # Scale Images

    # For testing: deterministic subset (use None for all images)
    n = None

    # Run bdsf on the images
    bdsf_out = bdsf_run(
        # Call dataset for Transform
        np.array([dataset[i] for i in range(n or len(dataset))]),
        out_folder=out_folder,
        names=dataset.names[:n],
        override=arguments.override,
        max_workers=96,
        ang_size=120,
    )
