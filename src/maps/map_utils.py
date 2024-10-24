import numpy as np
from skimage.transform import rescale
from scipy.stats import multivariate_normal
from scipy.ndimage import gaussian_filter, rotate


def scale_to_flux(img, flux):
    return img * flux / img.sum()


def upscale_image(img, current_size, target_size):
    assert (
        current_size < target_size
    ), f"Upscaling only! {current_size} -> {target_size}"
    scale_factor = target_size / current_size
    return rescale(img, scale_factor, anti_aliasing=True)


def gaussian_signal(size=1, angle=0, convolve=True, img_size=80):
    if angle != 0:
        out_size = img_size
        img_size = int(np.sqrt(2) * img_size)

    # Data
    x = np.linspace(-img_size // 2, img_size // 2, img_size)
    y = x.copy()
    X, Y = np.meshgrid(x, y)

    if size == 0.0:
        print("Zero size converted to single pixel")
        size = 1.5  # arcsec = 1 px

    # Set sigma in x and y direction
    match size:
        case int() | float():
            sigma_x = size
            sigma_y = size
        case tuple() | list():
            sigma_x, sigma_y = size

    # 1.5 arcsec/px, size in arcsec and sigma in px
    sigma_x /= 1.5
    sigma_y /= 1.5

    # Gaussian
    rv = multivariate_normal([0, 0], [[sigma_x, 0], [0, sigma_y]])

    # Probability Density
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    pd = rv.pdf(pos)

    # Rotate and crop to output size
    if angle != 0:
        pd = rotate(pd, angle, reshape=False)
        start, end = img_size // 2 - out_size // 2, img_size // 2 + out_size // 2
        pd = pd[start:end, start:end]

    # Normalize to 1
    pd /= pd.sum()

    # Convolve with beam
    if convolve:
        # Beam size: FWHM = 6 arcsec = 4 px
        pd = gaussian_filter(pd, sigma=4 / 2.355)

    return pd


def add_source_image(map_array, map_size_deg, source_arr, coords, centroid=None):

    # Convert coords to pixel coords
    map_size_px = map_array.shape[-2:]
    x, y = coords
    x_px, y_px = int((x + 0.5 * map_size_deg) * map_size_px[0]), int(
        (y + 0.5 * map_size_deg) * map_size_px[1]
    )

    # Set centroid coords
    x_c, y_c = (
        centroid
        if centroid is not None
        else (source_arr.shape[0] // 2, source_arr.shape[1] // 2)
    )

    # Determine slices for adding source to map
    x_slice = slice(x_px - x_c, x_px - x_c + source_arr.shape[0])
    y_slice = slice(y_px - y_c, y_px - y_c + source_arr.shape[1])

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

    # Add source to map
    map_array[x_slice, y_slice] += source_arr
    return map_array
