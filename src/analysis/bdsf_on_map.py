from collections import OrderedDict
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

    # Load Mosaic files names
    d = {p.name: p / "mosaic-blanked.fits" for p in paths.MOSAIC_DIR.iterdir()}
    mosaic_dict = OrderedDict(sorted(d.items()))

    # Output parent directory
    out_parent = paths.ANALYSIS_PARENT / "mosaic_bdsf"

    # Run BDSF on first two maps
    for map_file in list(mosaic_dict.values())[:2]:

        # Create output directory
        out_dir = out_parent / map_file.parent.name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Copy map file
        map_file_out = out_dir / map_file.name
        if map_file_out.exists():
            map_file_out.unlink()
        map_file_out.write_bytes(map_file.read_bytes())

        # Custom settings
        user_settings = {
            "frequency": 143650000.0,
        }

        # Run dat thang
        bdsf_on_map(map_file_out, **user_settings)