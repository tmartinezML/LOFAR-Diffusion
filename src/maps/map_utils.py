import subprocess
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from skimage.transform import rescale
from scipy.stats import multivariate_normal
from scipy.ndimage import gaussian_filter, rotate

import utils.paths as paths


def get_image(
    in_map,
    get_wcs=True,
):

    # Parse input
    match in_map:
        # Case: Map name
        case str() if not "/" in in_map:
            file = paths.SKY_MAP_PARENT / in_map / f"{in_map}.fits"

        # Case: Full path as string
        case str() if "/" in in_map:
            file = Path(in_map)

        # Case: Path object
        case Path():
            file = in_map

        # Anything else: Invalid input
        case _:
            raise ValueError(f"Invalid input: {in_map}")

    # Check file validity
    if not file.exists():
        raise FileNotFoundError(f"File not found: {file}")
    elif not (file.is_file() and file.suffix == ".fits"):
        raise ValueError(f"Invalid file: {file}")

    # Load data
    with fits.open(file) as hdul:
        image = hdul[0].data
        wcs = WCS(hdul[0].header, naxis=2) if get_wcs else None

    return image, wcs


def run_command_with_logging(logger, cmd, log_file, **kwargs):
    logger.info(f"Running command: {' '.join(cmd)}")
    with (
        subprocess.Popen(
            cmd,
            shell=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            **kwargs,
        ) as p,
        open(log_file, "w") as log,
    ):

        while p.poll() is None:
            line = p.stdout.readline()
            log.write(line)
            print(line, end="")  # Not sure if this will lead to double printing

        p.wait()

    return p


def process_compact_source(i, source, arcsec_per_px=1.5):
    try:
        # Generate source array & scale to flux
        source_arr = gaussian_signal(
            size=source["size"],
            angle=source.angle,
            convolve=True,
        )
        # Scale to flux
        source_arr *= (1 / source_arr.sum()) * source.flux
        return i, source_arr, False

    except Exception as e:
        return i, None, str(e)


def lofar_num2nu(num, station, n_chan=2, nu_clk=200.0e6):
    """
    Get channel freq in MHz from LOFAR SB
    Parameters
    ----------
    num: SB number
    station: str, LBA or HBA
    n_chan: int, number of channels
    nu_clk: clock frequency

    Returns
    -------
    freq : (4,) array of floats, frequencies of channels
    ref_freq: float, reference freq.
    delta_nu: float, bandwidth of channel
    """
    if nu_clk == 200e6:
        SBband = 195312.5
    elif nu_clk == 160e6:
        SBband = 156250.0
    n = 1 if station == "LBA" else 2
    nu_0 = nu_clk * (num / 1024 + (n - 1) / 2)  # ref freq of SB
    delta_nu = nu_clk / (1024 * n_chan)  # difference between channels
    f = np.arange(n_chan) * delta_nu + nu_0 - delta_nu * (n - 1) / 2
    return f, np.mean(f), delta_nu


def beam_solid_angle(beam_size):
    """
    Calculate the solid angle of a beam, given its size, in squared input units.

    Parameters
    ----------
    beam_size : int | float | tuple | list
        Size of the beam in arcseconds. If a tuple or list is given, the
        first two elements are used as the major and minor axis of the beam.
        Otherwise, the beam is assumed to be circular.

    Returns
    -------
    float
        Solid angle of the beam in same unit (squared tho) as the input.
    """
    match beam_size:
        case int() | float():
            bb = beam_size**2
        case tuple() | list():
            bb = beam_size[0] * beam_size[1]

    return np.pi / (4 * np.log(2)) * bb


def get_image(file, get_wcs=True):
    hdul = fits.open(file)
    image = hdul[0].data
    if get_wcs:
        wcs = WCS(hdul[0].header, naxis=2)
        return image, wcs
    else:
        return image


def scale_to_flux(img, flux):
    return img * flux / img.sum()


def upscale_image(img, current_size, target_size):
    assert (
        current_size < target_size
    ), f"Upscaling only! {current_size} -> {target_size}"
    scale_factor = target_size / current_size
    return rescale(img, scale_factor, anti_aliasing=True)


def gaussian_signal(size=1, angle=0, convolve=True, img_size=None):

    # Set source size (= 2*FWHM of gaussian) in arcsec
    if size == 0.0:
        # print("Zero size converted to single pixel")
        size = 1.5  # arcsec = 1 px

    # Set sigma in x and y direction
    # 2.355 is for FWHM to sigma conversion, and factor 2 is because we define
    # the relation as size = 2 * FWHM
    match size:
        # Single value, i.e. both directions are the same
        case int() | float():
            sigma_x = size / (2 * 2.355)
            sigma_y = size / (2 * 2.355)
        # Tuple or list, i.e. different values for x and y
        case tuple() | list():
            sigma_x, sigma_y = (s / (2 * 2.355) for s in size)
        case _:
            raise TypeError(f"Invalid size type: {type(size)} ({size})")

    # Set image size
    if img_size is None:
        img_size = np.ceil(5 * max(sigma_x, sigma_y)).astype(int)  # 5 Sigma in px
        img_size = max(img_size, 20)  # Minimum size of 20 pixel to easily fit beam.

    # If the angle is not 0, we need to rotate the image, therefore the generated
    # image will temporarily have the side length of the diagonal of the output image,
    # so we can safely rotate it without losing any information.
    if angle != 0:
        out_size = img_size
        img_size = (np.sqrt(2) * img_size).astype(int)

    # Prepare meshgrid, i.e. x-y data
    x = np.linspace(-img_size // 2, img_size // 2, img_size)
    y = x.copy()
    X, Y = np.meshgrid(x, y)

    # 1.5 arcsec/px, size in arcsec and sigma in px
    sigma_x /= 1.5
    sigma_y /= 1.5

    # 2D Gaussian model
    rv = multivariate_normal([0, 0], [[sigma_x**2, 0], [0, sigma_y**2]])

    # Flux density = Probability Density
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    img = rv.pdf(pos)

    # Rotate and crop to output size
    if angle != 0:
        img = rotate(img, angle, reshape=False)
        start, end = img_size // 2 - out_size // 2, img_size // 2 + out_size // 2
        img = img[start:end, start:end]

    # Convolve with beam
    if convolve:
        # Beam size: FWHM = 6 arcsec = 4 px
        img = gaussian_filter(img, sigma=4 / 2.355, mode='constant')

    # If the size is really small, we migth get a zero image.
    # In that case, we make an image with the center pixel set to 1.
    # Normalization and convolution happens afterwards.
    # Note: In some rare cases, it happens that the sum is not exactly zero,
    # but becomes zero after convolution. This is why we check for the sum
    # only after the convolution and possibly convolve again.
    if img.sum() == 0:
        img[img_size // 2, img_size // 2] = 1
        if convolve:
            img = gaussian_filter(img, sigma=4 / 2.355)

    # Normalize to 1
    img /= img.sum()

    return img


def add_source_image(map_array, map_size_deg, source_arr, coords, centroid=None):

    slices, source_arr = get_source_slice(
        coords, source_arr, map_array, map_size_deg, centroid=centroid
    )

    # Add source to map
    map_array[*slices] += source_arr
    return map_array


def get_source_slice(
    coords, source_arr, map_arr, map_size_deg, centroid=None, truncate_edge=True
):
    map_size_px = map_arr.shape[-2:]

    # Convert map coords to pixel coords
    x_px, y_px = coord2pix(coords, map_size_deg, map_size_px)

    # Set centroid coords (relative to source array)
    x_c, y_c = (
        (np.round(c).astype(int) for c in centroid)
        if centroid is not None
        else (source_arr.shape[0] // 2, source_arr.shape[1] // 2)
    )

    # Determine slices
    x_slice = slice(x_px - x_c, x_px - x_c + source_arr.shape[0])
    y_slice = slice(y_px - y_c, y_px - y_c + source_arr.shape[1])

    if truncate_edge:
        # Check if source is within map, otherwise correct slice to fit
        # and reduce source_arr accordingly
        if x_slice.start < 0:
            source_arr = source_arr[-x_slice.start :, :]
            x_slice = slice(0, x_slice.stop)
        if x_slice.stop > map_size_px[0]:
            source_arr = source_arr[: map_size_px[0] - x_slice.stop, :]
            x_slice = slice(x_slice.start, map_size_px[0])
        if y_slice.start < 0:
            source_arr = source_arr[:, -y_slice.start :]
            y_slice = slice(0, y_slice.stop)
        if y_slice.stop > map_size_px[1]:
            source_arr = source_arr[:, : map_size_px[1] - y_slice.stop]
            y_slice = slice(y_slice.start, map_size_px[1])

    return (x_slice, y_slice), source_arr


def coord2pix(coords, map_size_deg, map_size_px):
    match map_size_px:
        case int() | float():
            map_size_px = (map_size_px, map_size_px)
        case tuple() | list():
            map_size_px = map_size_px
        case _:
            raise ValueError(f"Invalid map_size_px dtype: {type(map_size_px)}")
    x, y = coords

    x_px = np.round((x / map_size_deg + 0.5) * map_size_px[0]).astype(int)
    y_px = np.round((y / map_size_deg + 0.5) * map_size_px[1]).astype(int)
    return x_px, y_px


def pix2coord(pix, map_size_deg, map_size_px):
    match map_size_px:
        case int() | float():
            map_size_px = (map_size_px, map_size_px)
        case tuple() | list():
            map_size_px = map_size_px
        case _:
            raise ValueError(f"Invalid map_size_px dtype: {type(map_size_px)}")
    x, y = pix
    x_deg = x / map_size_px[0] * map_size_deg - 0.5 * map_size_deg
    y_deg = y / map_size_px[1] * map_size_deg - 0.5 * map_size_deg
    return x_deg, y_deg


def make_fits_header(
    arcsec_per_px,
    map_size_px,
):
    header_cards = {
        # These are always the same
        "BUNIT": "Jy",
        "WCSAXES": 2,
        "CTYPE1": "RA---SIN",
        "CTYPE2": "DEC--SIN",
        "CUNIT1": "deg",
        "CUNIT2": "deg",
        "RADESYS": "ICRS",
        "EQUINOX": 2000.0,
        "LONPOLE": 180.0,
        "LATPOLE": 0.0,
        # For now, these will also be the same. Might make them variable later.
        "CTYPE3": "FREQ",
        "CUNIT3": "Hz",
        "CRVAL3": 143650000.0,
        "CDELT3": 48000000.0,
        "CRVAL1": 0.031250,
        "CRVAL2": 23.395251,
        # Those depend on the map
        "CDELT1": -arcsec_per_px / 3600,
        "CDELT2": arcsec_per_px / 3600,
        "NAXIS1": map_size_px,
        "NAXIS2": map_size_px,
        "CRPIX1": map_size_px // 2,
        "CRPIX2": map_size_px // 2,
    }
    header = fits.Header()
    header.update(header_cards)
    return header
