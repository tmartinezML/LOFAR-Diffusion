import numpy as np
from astropy.io import fits
from casacore.tables import table


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
