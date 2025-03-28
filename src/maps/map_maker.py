import ast
import warnings
from pathlib import Path
from configparser import ConfigParser
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import h5py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from astropy.io import fits
from astropy.table import Table
from scipy.stats import rv_histogram
from skimage.draw import disk
from skimage.transform import resize
from skimage.measure import regionprops_table

import utils.logging
import utils.paths as paths
import maps.map_utils as mputil
import model.model_utils as mdutil
import model.sampler as smplr
from data.cutouts import save_images_h5py
from data.datasets import parse_dset_path
from maps.estimate_npix import EstimateNpix
from maps.map_utils import process_compact_source
from data.segment import get_sample_mask, circular_mask

# TODO:
# - Add type hints & docstrings
# - Review variable names


class MapMaker:
    def __init__(
        self,
        *,
        map_name,
        model_name="Prototypes_Model_SizeCond",
        map_size_deg=5,
        trecs_cat_file=None,
        dset="prototypes",
        img_size=80,
        arcsec_per_px=1.5,
        max_sampling_size=80,
        min_flux_Jy=0,
        max_flux_Jy=2,  # Jy
        sampler_settings={"n_devices": 2},
    ):
        # Logger
        self.logger = utils.logging.get_logger(self.__class__.__name__)

        # Set map name and output directory
        self.map_name = map_name
        self.out_dir = paths.SKY_MAP_PARENT / map_name
        self.out_dir.mkdir(parents=False, exist_ok=True)
        self.trecs_dir = self.out_dir / "trecs"
        self.min_flux = min_flux_Jy
        self.max_flux = max_flux_Jy

        # Map parameters
        self.map_size_deg = map_size_deg
        self.arcsec_per_px = arcsec_per_px
        self.map_size_px = int(map_size_deg * 3600 / arcsec_per_px)

        # Initialize empty map array
        self.map_array = None
        self.reset_map_array()

        # Diffusion Model and sampling parameters
        self.model_name = model_name
        self.img_size = img_size
        self.max_sampling_size = max_sampling_size
        self.sampler_settings = sampler_settings

        # Extended sources, stored in lists because of different sizes
        self.ext_data = {
            "images": [],
            "masks": [],
        }
        self.ext_df = pd.DataFrame(
            columns=[
                "x_coord",
                "y_coord",
                "flux",
                "size",
                "context_size",
                "centroid-0",
                "centroid-1",
                "feret_diameter_max",
            ]
        )

        # Compact sources
        self.comp_images = None
        self.comp_df = pd.DataFrame(
            columns=["x_coord", "y_coord", "flux", "size", "angle"]
        )

        # This will store TRECS catalog path and dataset path
        self.input_data = {}

        # Read in T-RECS catalog if passed
        if trecs_cat_file is not None:
            self.read_TRECS(trecs_cat_file)

        # Get model size distribution if dataset is passed
        self.model_size_distribution = None, None
        if dset is not None:
            self.input_data["dset"] = str(dset)
            self.get_model_size_distribution()

        self.logger.info("MapMaker initialized.")

    @staticmethod
    def load(in_file):
        match in_file:
            case str():
                # Assume file name
                file_name = in_file
                in_file = paths.SKY_MAP_PARENT / f"{file_name}/{file_name}.h5"
            case Path():
                # Assume file object
                file_name = in_file.stem
            case _:
                raise ValueError(f"Invalid data type for input file: {type(in_file)}")

        # Read data arrays and attributes first
        with h5py.File(in_file, "r") as f:
            # Attributes
            map_size_deg = f.attrs["map_size_deg"]
            model_name = f.attrs["model_name"]
            mm = MapMaker(
                map_name=file_name,
                map_size_deg=map_size_deg,
                model_name=model_name,
                dset=None,
            )

            mm.give_name(file_name)

            mm.logger.info(f"Reading MapMaker instance from\n\t{in_file}...")
            mm.img_size = f.attrs["img_size"]
            mm.arcsec_per_px = f.attrs["arcsec_per_px"]
            mm.max_sampling_size = f.attrs["max_sampling_size"]
            mm.input_data = ast.literal_eval(f.attrs["input_data"])
            mm.sampler_settings = ast.literal_eval(f.attrs["sampler_settings"])
            mm.min_flux = f.attrs.get("min_flux", 0)

            # Data arrays
            mm.map_array = f["sky_map"][:]
            mm.model_size_distribution = (
                f["model_size_distribution/counts"][:],
                f["model_size_distribution/bins"][:],
            )

            # Variable length image data:
            def read_var_len_data(dset_name):
                data = f[dset_name][:]
                shapes = f[dset_name + "_shapes"][:]
                return [arr.reshape(shape) for arr, shape in zip(data, shapes)]

            mm.comp_images = read_var_len_data("compact_sources")
            mm.ext_data["images"] = read_var_len_data("extended_sources")
            mm.ext_data["masks"] = read_var_len_data("extended_source_masks")

        # Read metadata
        mm.comp_df = pd.read_hdf(in_file, key="compact_sources_metadata")
        mm.ext_df = pd.read_hdf(in_file, key="extended_sources_metadata")

        mm.logger.info("MapMaker instance read.")
        return mm

    def plot_map(self, scale_fn=lambda x: np.tanh(7.5 * x)):

        # Scale map
        scaled_map = scale_fn(self.map_array)

        # Plot map
        fig, ax = plt.subplots(figsize=(9, 9))
        plt.colorbar(ax.imshow(scaled_map, origin="lower"), fraction=0.046, pad=0.04)
        ax.axis("off")
        fig.show()
        return

    def save(self, file_name=None, override=False):
        # Set and check: file name, out file, override
        if file_name is None:
            if not self._check_hasname():
                return
        else:
            self.give_name(file_name)

        self.save_to_hdf(override=override)
        self.save_to_fits(override=override)

    def give_name(self, file_name):
        if self.map_name is not None:
            self.logger.warning(f"Changing file name {self.map_name} with {file_name}.")
        self.map_name = file_name
        self.out_dir = paths.SKY_MAP_PARENT / file_name

    def _check_override(self, out_file, override):
        if out_file.exists():
            if override:
                self.logger.warning(f"Overwriting existing file {out_file}.")
                out_file.unlink()
                return False
            else:
                self.logger.warning(
                    f"File {out_file} already exists. Set override=True to overwrite."
                )
                return True

    def _check_hasname(self):
        if not hasattr(self, "map_name"):
            self.logger.warning("No file name set - aborting. Use give_name() first.")
            return False
        return True

    def prepare_TRECS(self):
        # TODO: make t-recs dir class attribute??
        self.trecs_dir.mkdir(parents=False, exist_ok=True)

        # Write frequency file
        with open(self.trecs_dir / "frequency_list.dat", "w") as f:
            f.write("# Frequencies in MHz\n")
            f.write("144\n")

        # Read default parameter file
        conf = ConfigParser(inline_comment_prefixes=("#", ";"))
        conf.optionxform = str  # Preserve case
        # Read with a default section title. This is a workaround so we can use
        # ConfigParser, which requires section titles that T-RECS parameter
        # files don't use.
        with open(paths.MAP_DEFAULTS / "TRECS_parameter_file.ini", "r") as file:
            conf.read_string("[DEFAULT]\n" + file.read())

        # Set parameters
        conf["DEFAULT"]["sim_side"] = str(self.map_size_deg)
        conf["DEFAULT"]["seed"] = str(np.random.randint(1, 1e2))

        # Write the modified content back to a file without the 'DEFAULT' section title
        with open(self.trecs_dir / "parameter_file.ini", "w") as file:
            file.write('# For explanations, see default TRECS parameter file."\n')
            for key in conf["DEFAULT"]:
                file.write(f"{key} = {conf['DEFAULT'][key]}\n")

        # Write shell script for execution
        with open(self.trecs_dir / "trecs_run.sh", "w") as f:
            f.write(
                "#!/bin/bash\n"
                "export PATH=$PATH:/hs/fs08/data/group-brueggen/tmartinez/trecs/bin\n"
                "export LD_LIBRARY_PATH=/hs/fs08/data/group-brueggen/tmartinez/software/cfitsio/lib\n"
                f"cd {self.trecs_dir}\n"
                f"trecs -c -w -p parameter_file.ini\n"
            )

    def run_TRECS(self):
        # Run T-RECS
        self.logger.info("Running T-RECS...")

        cmd = [
            "sh",
            f'{str(self.trecs_dir / "trecs_run.sh")}',
        ]

        p = mputil.run_command_with_logging(
            self.logger,
            cmd,
            self.trecs_dir / "trecs_run.log",
        )
        return p

    def save_to_hdf(self, file_name=None, override=False):
        # Set and check: file name, out file, override
        if file_name is None:
            if not self._check_hasname():
                return
        else:
            self.give_name(file_name)
        out_file = self.out_dir / f"{self.map_name}.h5"
        if self._check_override(out_file, override):
            return

        self.logger.info(f"Saving MapMaker instance to\n\t{out_file}")
        if not out_file.parent.exists():
            self.logger.info(f"Creating directory\n\t{out_file.parent}")
            out_file.parent.mkdir(parents=True)

        # Save map
        save_images_h5py(
            self.map_array,
            out_file,
            dset_name="sky_map",
        )

        # Save compact sources
        save_images_h5py(
            self.comp_images,
            out_file,
            dset_name="compact_sources",
            dtype=h5py.vlen_dtype(self.comp_images[0].dtype),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
            self.comp_df.to_hdf(out_file, key="compact_sources_metadata")

        # Save extended sources
        save_images_h5py(
            self.ext_data["images"],
            out_file,
            dset_name="extended_sources",
            dtype=h5py.vlen_dtype(self.ext_data["images"][0].dtype),
        )
        save_images_h5py(
            self.ext_data["masks"],
            out_file,
            dset_name="extended_source_masks",
            dtype=h5py.vlen_dtype(self.ext_data["masks"][0].dtype),
        )
        self.ext_df.to_hdf(out_file, key="extended_sources_metadata")

        # Save model size distribution
        if self.model_size_distribution[0] is not None:
            # This actually works for general shape arrays
            save_images_h5py(
                np.array(self.model_size_distribution[0]),
                out_file,
                dset_name="model_size_distribution/counts",
            )
            save_images_h5py(
                np.array(self.model_size_distribution[1]),
                out_file,
                dset_name="model_size_distribution/bins",
            )

        # Save all other attributes
        with h5py.File(out_file, "a") as f:
            f.attrs["map_size_deg"] = self.map_size_deg
            f.attrs["map_size_px"] = self.map_size_px
            f.attrs["arcsec_per_px"] = self.arcsec_per_px
            f.attrs["img_size"] = self.img_size
            f.attrs["max_sampling_size"] = self.max_sampling_size
            f.attrs["model_name"] = self.model_name
            f.attrs["input_data"] = str(self.input_data)
            f.attrs["sampler_settings"] = str(self.sampler_settings)
            f.attrs["min_flux"] = self.min_flux

        self.logger.info("MapMaker saved.")

    def save_to_fits(self, chan_dim=False, file_name=None, override=False):
        # Set and check: file name, out file, override
        if file_name is None:
            if not self._check_hasname():
                return
        else:
            self.give_name(file_name)

        # Set out file name. If chan_dim is True, add '_chanDim' to file name.
        out_file = (
            self.out_dir
            / f"{self.map_name + '_chanDim' if chan_dim else self.map_name}.fits"
        )
        if self._check_override(out_file, override):
            return

        self.logger.info(f"Saving map data to\n\t{out_file}")
        header = mputil.make_fits_header(
            arcsec_per_px=self.arcsec_per_px,
            map_size_px=self.map_size_px,
        )
        hdu = fits.PrimaryHDU(
            header=header,
            data=np.expand_dims(self.map_array, 0) if chan_dim else self.map_array,
        )
        hdu.writeto(out_file, overwrite=True, output_verify="fix")
        self.logger.info("Map data saved.")

    def save_telsim_output(self):
        # The telescope simulator requires the following:
        # - The map in Jy/pixel with 2D array shape (see self.save_to_fits())
        # - The same map but with 3D array, with the first axis being the channel axis
        # - A mask with ddf shape

        # Save map data as fits and hdf
        self.save()

        # Save map data with channel dimension
        self.save_to_fits(chan_dim=True)

        # Mask with ddf shape
        self.make_threshold_mask(sensitivity=5e-5, save=True, mask_size="ddf")

    def read_TRECS(self, trecs_cat_file=None):
        if trecs_cat_file is None:
            trecs_cat_file = self.trecs_dir / "catalogue_continuum_wrapped.fits"
            if not trecs_cat_file.exists():
                raise FileNotFoundError(f"File not found: {trecs_cat_file}")

        # Read in T-RECS catalog
        self.logger.info(f"Reading T-RECS catalog from\n\t{trecs_cat_file}...")
        trecs_df = Table.read(
            trecs_cat_file, hdu=1, unit_parse_strict="silent"
        ).to_pandas()
        self.input_data["trecs_cat_file"] = str(trecs_cat_file)

        # Classes:
        # 1 - 3: SFGs
        # 4: FSRQ, 5: BL-Lac, 6: SS-AGN
        sfg_flag = trecs_df["RadioClass"].values < 4
        # Extended sources must be SS-AGN (class 6) and larger than one pixel (1.5 arcsec).
        # Anything else is considered compact.
        compact_flag = (trecs_df["RadioClass"].values < 6) | (trecs_df["size"] <= 1.5)
        comp_df, ext_df = (
            trecs_df[compact_flag],
            trecs_df[~compact_flag],
        )

        self.logger.info("Extracting values...")

        # Fill extended sources df
        self.ext_df["TRECS_index"] = ext_df.index.to_numpy()
        self.ext_df["x_coord"] = ext_df["x_coord"].values
        self.ext_df["y_coord"] = ext_df["y_coord"].values
        self.ext_df["flux"] = ext_df["I144"].values * 1e-3  # mJy to Jy

        # Fill compact sources df
        self.comp_df["TRECS_index"] = comp_df.index.to_numpy()
        self.comp_df["x_coord"] = comp_df["x_coord"].values
        self.comp_df["y_coord"] = comp_df["y_coord"].values
        self.comp_df["flux"] = comp_df["I144"].values * 1e-3  # mJy to Jy
        self.comp_df["size"] = comp_df["size"].values
        b_maj_min = np.array(list(zip(comp_df["bmaj"], comp_df["bmin"])), dtype="f,f")
        sfg_flag = comp_df["RadioClass"].values < 4
        self.comp_df.loc[sfg_flag, "size"] = b_maj_min[sfg_flag]
        self.comp_df["angle"] = comp_df["PA"].values
        self.comp_df.loc[~sfg_flag, "angle"] = 0

        # Filter sources by flux
        self.logger.info(
            f"Filtering for flux between {self.min_flux:.1e} and {self.max_flux:.1e} Jy..."
        )
        # Remove sources with flux below min_flux
        self.comp_df = self.comp_df[(cmp_flg := self.comp_df["flux"] >= self.min_flux)]
        self.ext_df = self.ext_df[(ext_flg := self.ext_df["flux"] >= self.min_flux)]
        dropouts_min = (~cmp_flg).sum() + (~ext_flg).sum()

        # Remove sources with flux above max_flux
        self.comp_df = self.comp_df[(cmp_flg := self.comp_df["flux"] <= self.max_flux)]
        self.ext_df = self.ext_df[(ext_flg := self.ext_df["flux"] <= self.max_flux)]
        dropouts_max = (~cmp_flg).sum() + (~ext_flg).sum()
        self.logger.info(
            f"Filtered {dropouts_min} compact sources below min_flux and {dropouts_max} above max_flux."
        )

        self.logger.info("T-RECS catalog read.")

    def get_model_size_distribution(self, dset=None, bins=100):
        dset = dset or self.input_data["dset"]
        dset_path = parse_dset_path(dset)
        self.logger.info(f"Extracting model size distribution from\n\t{dset_path}...")
        mask_metadata = pd.read_hdf(dset_path, key="mask_metadata")
        self.model_size_distribution = np.histogram(
            mask_metadata["feret_diameter_max"], bins=bins, density=True
        )
        self.logger.info("Model size distribution extracted.")
        return

    def save_masked_map(self, mask_file):
        if not self._check_hasname():
            return

        mask = fits.getdata(mask_file).squeeze()
        assert (
            mask.shape == self.map_array.shape
        ), "Mask shape does not match map shape."
        masked_map = self.map_array * mask

        out_file = self.out_dir / f"{self.map_name}_masked.fits"
        self.logger.info(f"Saving masked map data to\n\t{out_file}")
        header = mputil.make_fits_header(
            arcsec_per_px=self.arcsec_per_px,
            map_size_px=self.map_size_px,
        )
        header["MASKFILE"] = str(mask_file)
        hdu = fits.PrimaryHDU(header=header, data=masked_map)
        hdu.writeto(out_file, overwrite=True, output_verify="fix")
        self.logger.info("Masked map data saved.")
        return

    def make_threshold_mask(
        self,
        sensitivity=5e-5,  # Jy/beam
        peak_flux_threshold=None,
        save=True,
        mask_size=None,
    ):
        if sensitivity is not None and peak_flux_threshold is None:
            beam_size = 6  # arcsec (fixed for now)
            beam_angle = mputil.beam_solid_angle(beam_size)
            peak_flux_threshold = (
                sensitivity * self.arcsec_per_px**2 / beam_angle
            )  # Jy/pixel
            self.logger.info(
                f"Creating mask with sensitivity {sensitivity} Jy/beam = {peak_flux_threshold} Jy/pixel..."
            )
            file_suffix = f"{sensitivity:.2e}JyBeam^-1"

        elif peak_flux_threshold is not None:
            self.logger.info(
                f"Creating mask with peak flux threshold {peak_flux_threshold} Jy/pixel..."
            )
            file_suffix = f"{peak_flux_threshold:.2e}JyPx^-1"

        else:
            msg = "No sensitivity or peak flux threshold provided. Aborting."
            self.logger.error(msg)
            raise ValueError(msg)

        mask = self.map_array >= peak_flux_threshold

        match mask_size:
            case None | "ddf":
                self.logger.info("Estimating mask size for DDF...")
                mask_size = EstimateNpix(self.map_size_px)[0]
                file_suffix += "_ddfSize"

            case "model":
                mask_size = self.map_size_px
                file_suffix += "_modelSize"

            case int():
                file_suffix += f"_{mask_size}px"

            case _:
                msg = f"Invalid keyword argument for mask size: {mask_size} (of type {type(mask_size)}). Aborting."
                self.logger.error(msg)
                raise ValueError(msg)

        self.logger.info(
            f"Mask size: {mask_size} px (from map with {self.map_size_px} px)."
        )
        mask = resize(mask, output_shape=(mask_size,) * 2).astype(bool).astype(int)
        # Bring dimensions to standard format
        mask = np.expand_dims(mask, axis=(0, 1))

        if save and self._check_hasname():
            out_file = (
                self.out_dir / f"{self.map_name}_thresholdMask_{file_suffix}.fits"
            )
            self.logger.info(f"Saving map data to\n\t{out_file}")
            hdu = fits.PrimaryHDU(mask)
            hdu.writeto(out_file, overwrite=True)
            self.logger.info("Map data saved.")

        return mask, out_file

    def make_object_mask(self, flux_threshold=None, mask_size=None, save=True):
        # Select compact sources
        comp_flag = self.comp_df["flux"] > flux_threshold
        comp_df = self.comp_df[comp_flag]
        # For every compact source, circular mask with 2* beam size radius,
        # i.e. 8 px for 1.5 arcsec/px
        comp_mask = circular_mask(shape=(self.img_size,) * 2, radius=8)

        # Select extended sources
        ext_flag = self.ext_df["flux"] > flux_threshold
        ext_df = self.ext_df[ext_flag]
        ext_masks = [self.ext_data["masks"][i] for i in np.where(ext_flag)[0]]

        # Create new all-sky mask
        if mask_size is not None:
            mask_size = mask_size
        else:
            self.logger.info("Estimating mask size for DDF...")
        mask_size = mask_size or EstimateNpix(self.map_size_px)[0]
        self.logger.info(
            f"Creating mask with {mask_size} px (from map with {self.map_size_px} px)."
        )
        all_sky_mask = np.zeros((mask_size,) * 2)

        # Add compact sources to mask
        for _, source in comp_df.iterrows():
            coords = source.x_coord, source.y_coord
            all_sky_mask = mputil.add_source_image(
                all_sky_mask, self.map_size_deg, comp_mask, coords
            )

        # Add extended sources to mask
        for mask, (_, source) in zip(ext_masks, ext_df.iterrows()):
            coords = source.x_coord, source.y_coord
            centroid = int(source["centroid-1"]), int(source["centroid-0"])
            all_sky_mask = mputil.add_source_image(
                all_sky_mask, self.map_size_deg, mask, coords, centroid=centroid
            )

        # Bring back to 0 or 1 values, since there might be overlap
        all_sky_mask = (all_sky_mask > 0).astype(int)

        # Bring dimensions to standard format
        all_sky_mask = np.expand_dims(all_sky_mask, axis=(0, 1))

        if save and self._check_hasname():
            out_file = (
                self.out_dir
                / f"{self.map_name}_objectMask_{str(flux_threshold).replace('.', 'p')}.fits"
            )
            self.logger.info(f"Saving map data to\n\t{out_file}")
            hdu = fits.PrimaryHDU(all_sky_mask)
            hdu.writeto(out_file, overwrite=True)
            self.logger.info("Map data saved.")

        return all_sky_mask

    def reset_map_array(self):
        self.map_array = np.zeros((self.map_size_px,) * 2)
        return

    def make_map(self, make_sources=True):
        self.logger.info("Starting map generation...")

        if (a := self.map_array) is not None and a.any():
            self.logger.warning("Map array is not empty. Resetting...")
            self.reset_map_array()

        if not make_sources:
            self.logger.info(
                "Skipping generation of sources. If no sources are present in the MapMaker instance, this will raise an error."
            )

        # Add compact & extended sources
        if make_sources:
            # TODO: Implement decision on whether to run parallel
            self.make_compact_sources_parallel()
        self.add_compact_sources()

        if make_sources:
            self.make_extended_sources()
        self.add_extended_sources()

        self.logger.info("Map generated.")
        return

    def make_compact_sources(self):
        nsrc = len(self.comp_df)

        # Place compact sources on map

        self.comp_images = []
        for i, (_, source) in tqdm(
            enumerate(self.comp_df.iterrows()),
            desc="Making compact sources",
            total=nsrc,
        ):
            # Generate source array
            source_arr = mputil.gaussian_signal(
                size=source["size"],
                angle=source.angle,
                convolve=True,
            )

            # Scale to flux
            source_arr *= (1 / source_arr.sum()) * source.flux

            # Add to image list
            self.comp_images.append(source_arr)

    def make_compact_sources_parallel(self):
        self.logger.info("Generating compact sources in parallel...")
        nsrc = len(self.comp_df)

        # Place compact sources on map
        self.comp_images = [None] * nsrc

        with ProcessPoolExecutor(max_workers=16) as executor:
            futures = [
                executor.submit(
                    process_compact_source,
                    i,
                    source,
                    self.arcsec_per_px,
                )
                for i, (_, source) in tqdm(
                    enumerate(self.comp_df.iterrows()),
                    total=nsrc,
                    desc="Submitting jobs",
                )
            ]

            with tqdm(total=nsrc, desc="Making compact sources") as pbar:
                for future in as_completed(futures):
                    try:
                        i, source_arr, error = future.result()
                        if error:
                            self.logger.error(f"Error processing source {i}: {error}")
                        if source_arr is not None:
                            self.comp_images[i] = source_arr
                        pbar.update(1)
                    except Exception as e:
                        self.logger.error(f"Error in future result: {e}")

        # Raise error if there are any Nones in the list
        if any([img is None for img in self.comp_images]):
            raise ValueError("Error generating compact sources - None in image list.")

    def add_compact_sources(self):
        if self.comp_images is None:
            self.logger.warning("No compact sources found.")
            return

        nsrc = len(self.comp_df)

        # Place compact sources on map
        for i, (_, source) in tqdm(
            enumerate(self.comp_df.iterrows()),
            desc="Adding compact sources",
            total=nsrc,
        ):
            if source.flux < self.min_flux:
                continue

            # Get pixel coordinates
            coords = source.x_coord, source.y_coord

            # Add source array to map
            self.add_source_image(self.comp_images[i], coords)

    def add_extended_sources(self):
        # Place AGNs on map such that centroid matches the catalog x, y position
        if len(self.ext_data["images"]) == 0:
            print("No image data found for extended sources.")
            return

        nsrc = len(self.ext_df)

        for i, (_, source) in tqdm(
            enumerate(self.ext_df.iterrows()),
            desc="Adding extended sources",
            total=nsrc,
        ):
            if source.flux < self.min_flux:
                continue

            # Get pixel coordinates & centroid
            coords = source.x_coord, source.y_coord
            centroid = (source["centroid-1"], source["centroid-0"])
            source_arr = self.ext_data["images"][i] * self.ext_data["masks"][i]

            # Add source array to map
            self.add_source_image(source_arr, coords, centroid=centroid)

    def make_extended_sources(self):
        self.logger.info("Generating extended sources...")

        # Get model size distribution to sample the sizes from
        if self.model_size_distribution[0] is None:
            self.logger.warning(
                "No size model distribution found for extended sources. Aborting."
            )
            return
        size_rvs = rv_histogram(self.model_size_distribution)

        # Filter sources by flux
        # EDIT: For now, we filter by flux only when adding the sources,
        # we will generate all sources in the trecs cat then filter them.
        # self.logger.info(f"Filtering for minimum flux {self.min_flux} Jy...")
        # ext_df_filtered = self.ext_df[self.ext_df["flux"] >= self.min_flux]
        ext_df = self.ext_df
        nsrc = len(ext_df)

        # Get sizes from distribution, will be used as sampling context
        sizes = size_rvs.rvs(size=nsrc)
        context = np.clip(sizes, 0, self.max_sampling_size).reshape(-1, 1)

        # Apply box-cox transform to sizes
        size_transform = mdutil.load_data_transforms(self.model_name)["mask_sizes"]
        context_tr = torch.Tensor(size_transform.transform(context))

        # Sample source images
        sampler = smplr.Sampler(return_steps=False, **self.sampler_settings)
        samples = sampler.quick_sample(
            model_name=self.model_name,
            context=context_tr,
            # timesteps=5,  # Debug
        ).squeeze()

        image_list = []

        # If necessary, upscale images. Append to image list.
        for i, img in enumerate(samples):
            if (target_size := sizes[i]) != (current_size := context.squeeze()[i]):
                img = mputil.upscale_image(img, current_size, target_size)
            image_list.append(img)

        # Get masks & analyze their properties
        self.logger.info("Calculating masks & properties...")
        masks = [get_sample_mask(img) for img in image_list]
        mask_regionprops = [
            regionprops_table(mask, properties=("centroid", "feret_diameter_max"))
            for mask in tqdm(masks, desc="Calculating region properties")
        ]
        for d in mask_regionprops:
            for k, v in d.items():
                d[k] = v[0]  # Convert arrays with one entry to scalars

        # Scale the image to its respective Flux
        self.logger.info("Scaling images to flux...")
        for i, (_, source) in tqdm(
            enumerate(ext_df.iterrows()),
            desc="Scaling images",
            total=nsrc,
        ):
            image_list[i] *= (1 / (image_list[i] * masks[i]).sum()) * source.flux

        # Set class attributes:
        self.logger.info("Setting class attributes...")
        self.ext_data["images"] = image_list
        self.ext_data["masks"] = masks
        self.ext_df["size"] = sizes
        self.ext_df["context_size"] = context.squeeze()
        self.ext_df.update(mask_regionprops)
        self.ext_df = self.ext_df.astype(float)

        self.logger.info("Extended sources generated.")
        return

    # Function for adding images to the map
    def add_source_image(self, source_arr, coords, centroid=None):
        return mputil.add_source_image(
            self.map_array,
            self.map_size_deg,
            source_arr,
            coords,
            centroid=centroid,
        )


def run_map_maker(
    *,
    map_name,
    map_size_deg=5,
    model_name="Prototypes_Model_SizeCond",
    dset="prototypes",
    sampler_settings={"n_devices": 2},
):
    # TODO: This hsould be a function in the MapMaker class, with an option to
    # force re-run T-RECS. Default should be to check if there is already
    # a T-RECS catalog, and use it if so.

    # Check for existing files to prevent override
    out_dir = paths.SKY_MAP_PARENT / map_name
    if (out_dir / f"{map_name}.h5").exists() or (out_dir / f"{map_name}.fits").exists():
        raise FileExistsError(f"Map {map_name} already exists. Aborting for safety.")

    # Initialize MapMaker
    mm = MapMaker(
        map_name=map_name,
        map_size_deg=map_size_deg,
        model_name=model_name,
        dset=dset,
        sampler_settings=sampler_settings,
        max_flux_Jy=2,
    )

    # Run T-RECS
    mm.prepare_TRECS()
    p = mm.run_TRECS()
    p.wait()

    # Read data
    mm.read_TRECS()

    # Make map
    mm.make_map()

    # Save map
    mm.save(map_name, override=True)

    # Save chan-dim image
    mm.save_to_fits(chan_dim=True, override=True)

    # Make map-sized mask
    _, mask_file = mm.make_threshold_mask(
        sensitivity=5e-5, save=True, mask_size="model"
    )

    # Save masked map
    mm.save_masked_map(mask_file)

    # Make ddf-sized mask
    mm.make_threshold_mask(sensitivity=5e-5, save=True, mask_size="ddf")


if __name__ == "__main__":

    # Settings
    map_name = "map_verif_v1"

    # Run MapMaker
    run_map_maker(map_name=map_name)
