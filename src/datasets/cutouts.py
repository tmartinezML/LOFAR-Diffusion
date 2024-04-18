import tempfile

import h5py
import numpy as np
from tqdm import tqdm
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord

from utils.paths import cast_to_Path, MOSAIC_DIR, CUTOUTS_DIR


def single_cutout(
    data,
    header,
    wcs,
    name,
    ra,
    dec,
    size_px=None,
    las=None,
    f_las=1.5,
    export_dir=None,
    mask_nan=True,
):
    """
    Create a cutout of the mosaic for a given source.
    """
    contain_nan = False

    # Set size of cutout
    pixel_size = 1.5  # arcsec
    assert (size_px is not None) or (las is not None)
    if size_px is None:
        if las == 0.0:
            las = 15
        elif las < 10:
            las = 10
        size_px = round(f_las * las / pixel_size)  # In pixels

    # Create cutout
    center = SkyCoord(ra=[ra], dec=[dec], unit="deg")
    cutout = Cutout2D(data, center, size_px, wcs=wcs, mode="strict")

    # Check for NaNs, replace with nanmin if mask_nan is True
    if np.isnan(cutout.data).any():
        contain_nan = True
        if mask_nan:
            print(
                f"Nan values present in {name} cutout.\n" " Nan values set to nanmin."
            )
            cutout.data[np.isnan(cutout.data)] = np.nanmin(cutout.data)

    # Export cutout as .fits file if export_dir is passed
    if export_dir is not None:
        header.update(cutout.wcs.to_header())
        fits.writeto(
            cast_to_Path(export_dir) / f"{name}.fits",
            cutout.data,
            header,
            overwrite=True,
        )

    return cutout.data, contain_nan


def cutout_from_catalog(catalog, ind, mask_nan=False, size_px=None, opt_c=True):
    """
    Create a cutout from the mosaic for a given source in catalog.
    """
    # Get mosaic data and WCS from fits file
    mosaic = catalog["Mosaic_ID"].iloc[ind]
    file_name = f"../data/mosaics_public/{mosaic}/mosaic-blanked.fits"
    with fits.open(file_name) as hdul:
        data = hdul[0].data
        header = hdul[0].header
        wcs = WCS(header)
    name = catalog["Source_Name"].iloc[ind]

    # Get optical coordinates if desired/available,
    # otherwise use mean coordinates.
    if opt_c:
        ra = catalog["optRA"].iloc[ind]
        dec = catalog["optDec"].iloc[ind]
    if not opt_c or np.isnan(ra):
        ra = catalog["RA"].iloc[ind]
        dec = catalog["DEC"].iloc[ind]

    # Get LAS (largest angular size)
    las = catalog["LAS"].iloc[ind]
    las_type = catalog["LAS_from"].iloc[ind]  # for printing
    print(f"Source with index {ind} has a LAS of" f"{las:.3f} arcsec from {las_type}.")

    # Create cutout
    return single_cutout(
        data, header, wcs, name, ra, dec, las=las, mask_nan=mask_nan, size_px=size_px
    )


def get_cutouts(
    catalog,
    size_px=None,
    f=1.5,
    opt_c=True,
    fname_comment="",
    export_dir=CUTOUTS_DIR,
    mosaic_dir=MOSAIC_DIR,
):
    """
    Create cutouts for all the sources in the catalogue.
    """
    # Create export directory
    mosaic_dir, export_dir = cast_to_Path(mosaic_dir), cast_to_Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    # Name for outfile:
    out_name = f"cutouts"
    out_name += f"_{size_px}p" if size_px is not None else f"_{f}las".replace(".", "p")
    out_name += "_optC" if opt_c else "_radioC"
    out_file = export_dir / f"{out_name}_{fname_comment}.hdf5"

    assert not out_file.exists(), f"{out_file} already exists. Aborted for safety."

    # Create columns in catalog for problems,
    # will be set to True if there is a problem
    catalog["Problem_cutout"] = False
    catalog["Nans_cutout"] = False

    # Loop through unique mosaic IDs in catalogue
    mosaic_ids = catalog["Mosaic_ID"].unique()
    images = []
    for mosaic in tqdm(mosaic_ids, desc="Looping through mosaics..."):

        # Filter catalogue for sources in mosaic
        catalog_mosaic = catalog[catalog["Mosaic_ID"] == mosaic]

        # Assign variables from catalogue:
        # Names
        names = catalog_mosaic["Source_Name"].values
        # Coordinates
        if opt_c:
            ras_opt = catalog_mosaic["optRA"].values
            decs_opt = catalog_mosaic["optDec"].values
            ras_mean = catalog_mosaic["RA"].values
            decs_mean = catalog_mosaic["DEC"].values
            ras = np.where(np.isnan(ras_opt), ras_mean, ras_opt)
            decs = np.where(np.isnan(decs_opt), decs_mean, decs_opt)
        else:
            ras = catalog_mosaic["RA"].values
            decs = catalog_mosaic["DEC"].values
        # LAS (largest angular size)
        lass = catalog_mosaic["LAS"].values

        # Get mosaic data and WCS from fits file
        with fits.open(mosaic_dir / f"{mosaic}/mosaic-blanked.fits") as hdu:
            data = hdu[0].data
            header = hdu[0].header
        wcs = WCS(header)

        # Loop through sources in mosaic and create cutouts
        for name, ra, dec, las in zip(names, ras, decs, lass):

            # Boolean mask for accessing source entries in catalogue
            source_mask = catalog["Source_Name"] == name

            try:
                # Create cutout
                img_data, contain_nan = single_cutout(
                    data,
                    header,
                    wcs,
                    name,
                    ra,
                    dec,
                    size_px=size_px,
                    las=las,
                    f_las=f,
                    export_dir=None,
                )
                images.append(img_data)

                # Set Nans_cutout to True if there are NaNs
                catalog.loc[source_mask, "Nans_cutout"] = contain_nan

            except Exception as exc:
                print(f"Problem with {name} cutout.\n{exc}")
                catalog.loc[source_mask, "Problem_cutout"] = True

    save_images_hpy5(
        np.array(images) if size_px is not None else images,
        out_file,
        dset_name="cutouts",
        src_names=None,
        attributes={
            "size_px": size_px,
            "f_las": f,
            "opt_c": opt_c,
            "comment": fname_comment,
        },
        dtype=h5py.vlen_dtype(images[0].dtype) if size_px is None else None,
    )
    catalog.to_hdf(
        out_file,
        key="catalog",
        format="table",
        mode="a",
    )

    return catalog, out_file


def save_images_hpy5(
    img_array, save_path, dset_name="images", src_names=None, attributes={}, dtype=None
):
    out_file = h5py.File(save_path, "a")

    # Create a dataset in the file
    match img_array:
        # Used for fixed size images
        case np.ndarray():
            img_dset = out_file.create_dataset(dset_name, data=img_array, dtype=dtype)

        # Used for variable size images
        case list():
            print("Creating variable size dataset.")
            dtype = h5py.vlen_dtype(img_array[0].dtype)
            img_dset = out_file.create_dataset(
                dset_name, (len(img_array),), dtype=dtype
            )
            shapes = []
            for i, img in enumerate(img_array):
                img_dset[i] = img.flatten()
                shapes.append(img.shape)
            img_dset.attrs["flattened"] = True
            out_file.create_dataset(
                f"{dset_name}_shapes",
                data=np.array(shapes),
            )

    # Add source names if passed:
    if src_names is not None:
        out_file.create_dataset(
            f"{dset_name}_names",
            data=src_names.astype(object),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )

    # Add attributes:
    for key, value in attributes.items():
        img_dset.attrs[key] = value

    out_file.close()