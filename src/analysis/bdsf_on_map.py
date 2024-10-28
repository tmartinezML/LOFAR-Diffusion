import bdsf

import utils.paths as paths

# To do:
# - Add logging
# - See if beam size is contained in fits header.


def bdsf_on_map(map_file, **user_settings):
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
    # BDSF default settings, as in LoTSS-DR2 paper.
    beam_size = 0.001667

    settings = {
        "atrous_do": True,  # Wavelet decomposition
        "thresh_isl": 4,
        "thresh_pix": 5,
        "adaptive_rms_box": True,
        "beam": (beam_size, beam_size, 0),  # TO DO: Get beam size from fits header
        "output_all": True,
    }

    # Update default settings with user settings
    settings.update(user_settings)

    # Load the map
    img = bdsf.process_image(str(map_file), **settings)

    return img


if __name__ == "__main__":
    map_file = paths.MOSAIC_DIR / "P202+42/mosaic-blanked.fits"
    out_dir = paths.ANALYSIS_PARENT / "mosaic_bdsf"

    user_settings = {
        "solnname": map_file.parent.name,  # Appended to output dir
        "outdir": out_dir,
    }

    bdsf_on_map(map_file, **user_settings)
