import tempfile
from collections import OrderedDict

import bdsf
import numpy as np
from pathlib import Path
from astropy.io import fits

import utils.paths as paths
from maps.map_utils import beam_solid_angle

# To do:
# - See if beam size is contained in fits header.
# - Add bdsf output folder to map parent

# BDSF default settings, as in LoTSS-DR2 paper.
beam_size = 0.001667  # 6 arcsec in deg
standard_settings = {
    "atrous_do": True,  # Wavelet decomposition
    "thresh_isl": 4,
    "thresh_pix": 5,
    "adaptive_rms_box": True,
    "beam": (beam_size, beam_size, 0),  # TO DO: Get beam size from fits header
    "output_all": True,
}


def save_output(bdsf_img, src_file):
    """
    Save the output of a BDSF run to a file.
    """
    # Save model
    bdsf_img.export_image(
        outfile=str(src_file.parent / f"{src_file.stem}.bdsf.model.fits"),
        clobber=True,
        img_type="gaus_model",
    )

    # Save catalog
    for cat_type in ["srl", "gaul"]:
        bdsf_img.write_catalog(
            outfile=str(src_file.parent / f"{src_file.stem}.bdsf.{cat_type}.fits"),
            format="fits",
            clobber=True,
            catalog_type=cat_type,
        )
    return


def bdsf_on_map(file_input, **user_settings):
    """
    Run BDSF on a map file with the given settings.

    Parameters
    ----------
    map_file : Path
        The map file to run BDSF on.
    **settings
        Additional settings to pass to BDSF.

    Returns
    -------
    BDSF object
        The BDSF object containing the extracted sources.
    """
    # Parse input
    match file_input:
        case str():
            # Path as string:
            if "/" in file_input:
                map_file = paths.cast_to_Path(file_input)

            # Map name:
            else:
                map_file = (
                    paths.SKY_MAP_PARENT
                    / file_input
                    / f"ddf/{file_input}.app.restored.fits"
                )

        case Path():
            # Path to file:
            map_file = file_input

    # Combine default and user settings
    settings = standard_settings.copy()
    settings.update(user_settings)

    # Process map (all output saved)
    img = bdsf.process_image(str(map_file), **settings)

    return img


def bdsf_on_model(
    file_input,
    out_dir=None,
    **bdsf_kwargs,
):
    """
    Run bdsf on a single sky model map. Will add a small amount of noise to the
    image before running bdsf. This requires a temporary file to be created,
    which pybdsf needs as input. Also, the image is converted from Jy/pixel to
    Jy/beam.
    """
    # Parse input to file Path object
    match file_input:
        case str():
            # Path as string:
            if "/" in file_input:
                img_file = paths.cast_to_Path(file_input)

            # Map name:
            else:
                img_file = paths.SKY_MAP_PARENT / file_input / f"{file_input}.fits"

        case Path():
            # Path to file:
            img_file = file_input

    if out_dir is None:
        out_dir = img_file.parent

    # Load image and header
    with fits.open(img_file) as hdul:
        img = hdul[0].data.squeeze()
        header = hdul[0].header

    # Convert from Jy/pixel to Jy/beam
    img *= beam_solid_angle(6) / 1.5**2

    # Add small amount of noise, otherwise sigma-clipping algorithm called
    # within bdsf.process_image (functions.bstat) might not converge.
    noise_scale = 5e-5 * 1e-3
    z = np.random.normal(0, scale=noise_scale, size=img.shape)
    img += z

    # Run bdsf
    img = bdsf_on_array(
        img,
        header,
        src_file=img_file.with_suffix(".bdsfNoise.fits"),
        atrous_do=False,  # Instabilities when used on model
        **bdsf_kwargs,
    )

    # Return bdsf image object
    return img


def bdsf_on_array(
    img,
    header,
    src_file=None,
    out_dir=None,
    **bdsf_kwargs,
):
    """
    Run bdsf on a single sky model map. This requires a temporary file to be created,
    which pybdsf needs as input.
    """
    if out_dir is None:
        out_dir = src_file.parent if src_file is not None else paths.ANALYSIS_PARENT

    if not out_dir.exists():
        raise FileNotFoundError(f"Output directory {out_dir} does not exist.")

    # Create fits file
    fits_file = src_file if src_file is not None else out_dir / "img_fits_model.fits"
    # Write the hdu to tmp fits file
    hdu = fits.PrimaryHDU(data=img, header=header)
    fits.HDUList([hdu]).writeto(str(fits_file), overwrite=True)

    settings = standard_settings.copy()
    settings.update(bdsf_kwargs)

    img = bdsf.process_image(
        fits_file,
        # out_dir=out_dir,
        **settings,
    )

    # Return bdsf image object
    return img


if __name__ == "__main__":

    map_name = "map_verif_v1"
    ddf_parent = "ddf"
    map_file = (
        paths.SKY_MAP_PARENT / map_name / ddf_parent / f"{map_name}.int.restored.fits"
    )

    # Run BDSF on the map
    img = bdsf_on_map(map_file)

    # Run BDSF on the model
    img = bdsf_on_model(map_name)
