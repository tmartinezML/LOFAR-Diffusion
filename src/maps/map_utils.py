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
