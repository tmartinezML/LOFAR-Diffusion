from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool

import pandas as pd
import numpy as np


import utils.paths as paths


cat_file = paths.LOFAR_RES_CAT
out_folder = paths.LOFAR_DATA_PARENT / "source_separations"
out_folder.mkdir(exist_ok=True)
cat = pd.read_csv(cat_file)


def separation(ra_dec_1, ra_dec_2):
    ra1, dec1 = [np.deg2rad(a) for a in ra_dec_1]
    ra2, dec2 = [np.deg2rad(a) for a in ra_dec_2]

    # Haversine formula
    sep = 2 * np.arcsin(
        np.sqrt(
            np.sin((dec2 - dec1) / 2) ** 2
            + np.cos(dec1) * np.cos(dec2) * np.sin((ra2 - ra1) / 2) ** 2
        )
    )
    return np.rad2deg(sep)


def get_radec(src):
    if pd.notna(src["optRA"]):
        ra, dec = src["optRA"], src["optDec"]
    else:
        ra, dec = src["RA"], src["DEC"]
    return ra, dec


def source_separations(mosaic):
    mosaic_dir = out_folder / mosaic
    mosaic_dir.mkdir(exist_ok=True)
    mosaic_cat = cat[cat["Mosaic_ID"] == mosaic].reset_index(drop=True)

    # Calculate pair-wise separations for all sources in mosaic,
    # Return df where rows and cols are source names and values are separations
    num_sources = len(mosaic_cat)
    separations = np.full((3, num_sources, num_sources), np.nan)
    source_names = mosaic_cat["Source_Name"].values

    for i, src1 in mosaic_cat.iterrows():
        ra_dec_1 = get_radec(src1)
        for j, src2 in mosaic_cat.iterrows():
            if i < j:
                ra_dec_2 = get_radec(src2)
                sep = separation(ra_dec_1, ra_dec_2)
                d_ra_dec = [
                    (a2 - a1 + 180) % 360 - 180 for a1, a2 in zip(ra_dec_1, ra_dec_2)
                ]
                separations[0, i, j] = sep
                separations[1:, i, j] = d_ra_dec

    # Convert the upper triangular matrix to a DataFrame
    sep_df = pd.DataFrame(separations[0], index=source_names, columns=source_names)
    sep_df.to_csv(mosaic_dir / f"source_separations_{mosaic}.csv")

    d_ra_df = pd.DataFrame(separations[1], index=source_names, columns=source_names)
    d_ra_df.to_csv(mosaic_dir / f"source_delta_ra_{mosaic}.csv")

    d_dec_df = pd.DataFrame(separations[2], index=source_names, columns=source_names)
    d_dec_df.to_csv(mosaic_dir / f"source_delta_dec_{mosaic}.csv")


def process_mosaic(mosaic):
    try:
        source_separations(mosaic)
    except Exception as e:
        print(f"Error processing mosaic {mosaic}: {e}")


def source_separations_catalog_parallel():
    mosaics = cat["Mosaic_ID"].unique()
    with Pool() as pool:
        for _ in tqdm(pool.imap_unordered(process_mosaic, mosaics), total=len(mosaics)):
            pass


if __name__ == "__main__":
    source_separations_catalog_parallel()
