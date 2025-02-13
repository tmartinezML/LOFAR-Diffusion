import numpy as np
import concurrent.futures
from tqdm import tqdm
from astropy.io import fits
from scipy.ndimage import gaussian_filter

import utils.logging
import utils.paths as paths
import maps.map_utils as mputil
from analysis.bdsf_on_map import bdsf_on_array

# Get logger
logger = utils.logging.get_logger("MajItAnalysis")


def process_iteration(
    i,
    model_file,
    resid_file,
    beam_model,
    smooth_beam_model,
    beam_sigma,
    out_dir,
    map_name,
):
    logger.info(f"Processing iteration {i+1}...")

    # Get model data
    model_arr = mputil.get_image(model_file, get_wcs=False)

    # Get residual data
    with fits.open(resid_file) as hdul:
        resid_arr = hdul[0].data
        header = hdul[0].header

    # Convolve model_arr with gaussian
    conv_model_arr = (
        gaussian_filter(model_arr, beam_sigma, mode="constant")
        * mputil.beam_solid_angle(6)
        / 1.5**2
        * np.sqrt(beam_model)
        / np.sqrt(smooth_beam_model)
    )

    # Make restored image
    restored = conv_model_arr + resid_arr / np.sqrt(smooth_beam_model)

    # Get mock src file for correct pybdsf output naming
    mock_src_file = out_dir / f"{map_name}.restored_{i+1}.fits"

    logger.info("Running bdsf...")
    # Run bdsf on restored image
    bdsf_on_array(
        restored,
        header,
        src_file=mock_src_file,
        quiet=True,
        ncores=16,
    )
    return i


def run(map_name, ddf_parent_name="ddf"):
    # Define folder names
    map_parent = paths.SKY_MAP_PARENT / map_name
    ddf_dir = map_parent / ddf_parent_name

    # Create output directory
    out_dir = map_parent / "bdsf_majit"
    out_dir.mkdir(exist_ok=True)

    # Get model files & identify number of major iterations
    get_iter = lambda f: int(f.name.split(".")[1][-2:])
    model_files = sorted(list(ddf_dir.glob(f"{map_name}.model*.fits")), key=get_iter)
    n_iter = len(model_files)

    # Get residual files
    resid_files = sorted(list(ddf_dir.glob(f"{map_name}.residual*.fits")))
    assert n_iter == len(resid_files), "Number of model and residual files do not match"

    # Get beam files & load beam model
    beam_file = ddf_dir / f"{map_name}.Norm.fits"
    smooth_beam_file = ddf_dir / f"{map_name}.SmoothNorm.fits"
    logger.info(f"Loading beam model...")
    beam_model = mputil.get_image(beam_file, get_wcs=False)
    smooth_beam_model = mputil.get_image(smooth_beam_file, get_wcs=False)

    # Get resroring beam sigma
    beam_FWHM = 4  # 6 arcec = 4 pixels
    beam_sigma = beam_FWHM / (2 * np.sqrt(2 * np.log(2)))

    # Process major iterations in parallel
    logger.info(f"Starting parallel processing of {n_iter} major iterations...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_iteration,
                i,
                model_files[i],
                resid_files[i],
                beam_model,
                smooth_beam_model,
                beam_sigma,
                out_dir,
                map_name,
            )
            for i in range(n_iter)
        ]
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=n_iter,
            desc="Parallel Processing",
        ):
            try:
                i = (
                    future.result()
                )  # This will raise any exceptions that occurred during processing
            except Exception as e:
                logger.error(f"Exception occurred: {e}")

    logger.info("Processing complete.")


if __name__ == "__main__":
    map_name = "map_5deg_v2"
    run(map_name)
