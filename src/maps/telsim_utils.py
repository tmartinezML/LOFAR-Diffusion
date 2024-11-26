import tempfile

import bdsf
import numpy as np
from astropy.io import fits
from casacore.tables import table

from maps.map_utils import beam_solid_angle


def get_ndim(fits_file):
    """
    Check the number of dimensions of a FITS file.
    """
    with fits.open(fits_file) as hdul:
        return hdul[0].data.ndim


def get_mean_freq(ms):
    with table(str(ms) + "/SPECTRAL_WINDOW", ack=False) as t:
        freqs = t.getcol("CHAN_FREQ")[0] * 1e-6  # MHz
    return np.mean(freqs)


def bdsf_on_model(
    img_file,
    out_dir=None,
    **bdsf_kwargs,
):
    """
    Run bdsf on a single image.
    """
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
    # The noise value is derived from a lower sensitivity limit of 0.05 mJy/beam
    noise_scale = 0.05 * 1e-3
    z = np.random.normal(0, scale=noise_scale, size=img.shape)
    img += z

    hdu = fits.PrimaryHDU(data=img, header=header)

    # Create a temporary file
    with tempfile.NamedTemporaryFile(
        prefix="tmp_model", suffix=".fits", dir=out_dir
    ) as f:

        # Write the hdu to tmp fits file
        fits.HDUList([hdu]).writeto(f.name, overwrite=True)

        kwargs = {
            "frequency": header["CRVAL3"],
            "thresh_isl": 3,
            "thresh_pix": 5,
            "mean_map": "const",
            "rms_map": False,
            "thresh": "hard",
            "quiet": False,
            "debug": True,
        }
        kwargs.update(bdsf_kwargs)

        beam_size = 0.001667  # 6 arcsec in deg

        img = bdsf.process_image(
            f.name,
            beam=(beam_size, beam_size, 0),
            **kwargs,
        )

    # Return bdsf image object
    return img
