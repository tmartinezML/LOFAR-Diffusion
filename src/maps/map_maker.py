import warnings

import h5py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from astropy.io import fits
from astropy.table import Table
from scipy.stats import rv_histogram
from skimage.measure import regionprops_table

import utils.logging
import utils.paths as paths
import maps.map_utils as mputil
import model.model_utils as mdutil
import model.sampler as smplr
from data.cutouts import save_images_h5py
from data.segment import get_sample_mask
from data.datasets import parse_dset_path

# TODO:
# - Add type hints & docstrings
# - Review variable names


class MapMaker:
    def __init__(
        self,
        *,
        map_size_deg,
        model_name,
        trecs_cat_file=None,
        dset=None,
        img_size=80,
        max_sampling_size=80,
    ):
        # Logger
        self.logger = utils.logging.get_logger(self.__class__.__name__)

        # Map parameters
        self.map_size_deg = map_size_deg
        self.map_size_px = int(map_size_deg * 3600 / 1.5)  # 1.5 arcsec per pixel
        self.map_array = np.zeros((self.map_size_px,) * 2)
        self.img_size = img_size
        self.max_sampling_size = max_sampling_size
        self.model_name = model_name

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
        self.model_size_distribution = None, None

        # Compact sources
        self.comp_images = None
        self.comp_df = pd.DataFrame(
            columns=["x_coord", "y_coord", "flux", "size", "angle"]
        )

        self.input_data = {}

        # Read in T-RECS catalog if passed
        if trecs_cat_file is not None:
            self.input_data["trecs_cat_file"] = trecs_cat_file
            self.read_TRECS(trecs_cat_file)

        # Get model size distribution if dataset is passed
        if dset is not None:
            self.input_data["dset"] = dset
            self.get_model_size_distribution(dset)

        self.logger.info("MapMaker initialized.")

    def from_hdf(file_name):
        in_file = paths.SKY_MAP_PARENT / f"{file_name}/{file_name}.h5"

        # Read data arrays and attributes first
        with h5py.File(in_file, "r") as f:
            # Attributes
            map_size_deg = f.attrs["map_size_deg"]
            model_name = f.attrs["model_name"]
            mm = MapMaker(
                map_size_deg=map_size_deg,
                model_name=model_name,
            )

            mm.logger.info(f"Reading MapMaker instance from\n\t{in_file}...")
            mm.img_size = f.attrs["img_size"]
            mm.max_sampling_size = f.attrs["max_sampling_size"]
            mm.input_data = f.attrs["input_data"]

            # Data arrays
            mm.map_array = f["sky_map"][:]
            mm.comp_images = f["compact_sources"][:]
            mm.model_size_distribution = (
                f["model_size_distribution/counts"][:],
                f["model_size_distribution/bins"][:],
            )

            # Variable length image data:
            def read_var_len_data(dset_name):
                data = f[dset_name][:]
                shapes = f[dset_name + "_shapes"][:]
                return [arr.reshape(shape) for arr, shape in zip(data, shapes)]

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
        plt.colorbar(
            ax.imshow(scaled_map, origin="lower"), fraction=0.046, pad=0.04
        )
        ax.axis("off")
        fig.show()
        return

    def save(self, file_name):
        self.save_to_hdf(file_name)
        self.save_to_fits(file_name)

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

    def save_to_hdf(self, file_name, override=False):
        out_file = paths.SKY_MAP_PARENT / f"{file_name}.h5"
        if self._check_override(out_file, override):
            return

        self.logger.info(f"Saving MapMaker instance to\n\t{out_file}...")

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
            f.attrs["img_size"] = self.img_size
            f.attrs["max_sampling_size"] = self.max_sampling_size
            f.attrs["model_name"] = self.model_name
            f.attrs["input_data"] = str(self.input_data)

        self.logger.info("MapMaker saved.")

    def save_to_fits(self, file_name, override=False):
        out_file = paths.SKY_MAP_PARENT / f"{file_name}.fits"
        if self._check_override(out_file, override):
            return
        self.logger.info(f"Saving map data to\n\t{out_file}...")
        hdu = fits.PrimaryHDU(self.map_array)
        hdu.writeto(out_file, overwrite=True)
        self.logger.info("Map data saved.")

    def read_TRECS(self, trecs_cat_file):
        # Read in T-RECS catalog
        self.logger.info(f"Reading T-RECS catalog from\n\t{trecs_cat_file}...")
        trecs_df = Table.read(
            trecs_cat_file, hdu=1, unit_parse_strict="silent"
        ).to_pandas()

        # Classes:
        # 1 - 3: SFGs
        # 4: FSRQ, 5: BL-Lac, 6: SS-AGN
        sfg_flag = trecs_df["RadioClass"].values < 4
        compact_flag = trecs_df["RadioClass"].values < 6
        comp_df, ext_df = (
            trecs_df[compact_flag],
            trecs_df[~compact_flag],
        )

        self.logger.info("Extracting values...")
        # Fill extended sources df
        self.ext_df["x_coord"] = ext_df["x_coord"].values
        self.ext_df["y_coord"] = ext_df["y_coord"].values
        self.ext_df["flux"] = ext_df["I144"].values

        # Fill compact sources df
        self.comp_df["x_coord"] = comp_df["x_coord"].values
        self.comp_df["y_coord"] = comp_df["y_coord"].values
        self.comp_df["flux"] = comp_df["I144"].values
        self.comp_df["size"] = comp_df["size"].values
        b_maj_min = np.array(list(zip(comp_df["bmaj"], comp_df["bmin"])), dtype="f,f")
        sfg_flag = comp_df["RadioClass"].values < 4
        self.comp_df.loc[sfg_flag, "size"] = b_maj_min[sfg_flag]
        self.comp_df["angle"] = comp_df["PA"].values
        self.comp_df.loc[~sfg_flag, "angle"] = 0

        self.logger.info("T-RECS catalog read.")

    def get_model_size_distribution(self, dset, bins=100):
        dset_path = parse_dset_path(dset)
        self.logger.info(f"Extracting model size distribution from\n\t{dset_path}...")
        mask_metadata = pd.read_hdf(dset_path, key="mask_metadata")
        self.model_size_distribution = np.histogram(
            mask_metadata["feret_diameter_max"], bins=bins, density=True
        )
        self.logger.info("Model size distribution extracted.")
        return

    def make_map(self):
        self.logger.info("Generating map...")

        # Add compact & extended sources
        self.add_compact_sources()
        self.make_extended_sources()
        self.add_extended_sources()

        self.logger.info("Map generated.")
        return

    def add_compact_sources(self):
        # Place compact sources on map
        self.comp_images = np.zeros((len(self.comp_df), self.img_size, self.img_size))
        for i, (_, source) in tqdm(
            enumerate(self.comp_df.iterrows()),
            desc="Adding compact sources",
            total=len(self.comp_df),
        ):

            # Get pixel coordinates
            coords = source.x_coord, source.y_coord

            # Generate source array & scale to flux
            source_arr = mputil.gaussian_signal(
                size=source.size,
                angle=source.angle,
                convolve=True,
                img_size=self.img_size,
            )
            self.comp_images[i] = source_arr
            source_arr *= source.flux  # Gaussian signal is normalized already

            # Add source array to map
            self.add_source_image(source_arr, coords)

    def add_extended_sources(self):
        # Place AGNs on map such that centroid matches the catalog x, y position
        if len(self.ext_data["images"]) == 0:
            print("No image data found for extended sources.")
            return

        for i, (_, source) in tqdm(
            enumerate(self.ext_df.iterrows()),
            desc="Adding extended sources",
            total=len(self.ext_df),
        ):

            # Get pixel coordinates & centroid
            coords = source.x_coord, source.y_coord
            centroid = int(source["centroid-1"]), int(source["centroid-0"])
            source_arr = self.ext_data["images"][i] * self.ext_data["masks"][i]

            # Scale source to flux
            source_arr = mputil.scale_to_flux(source_arr, source.flux)

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

        # Get sizes from distribution, will be used as sampling context
        sizes = size_rvs.rvs(size=len(self.ext_df))
        context = np.clip(sizes, 0, self.max_sampling_size).reshape(-1, 1)

        # Apply box-cox transform to sizes
        size_transform = mdutil.load_data_transforms(self.model_name)["mask_sizes"]
        context_tr = torch.Tensor(size_transform.transform(context))

        # Sample source images
        sampler = smplr.Sampler(return_steps=False)
        samples = sampler.quick_sample(
            model_name=self.model_name,
            context=context_tr,
            # timesteps=5,  # Debug
        ).squeeze()

        # If necessary, upscale images. Append to image list.
        for i, img in enumerate(samples):
            if (target_size := sizes[i]) != (current_size := context.squeeze()[i]):
                img = mputil.upscale_image(img, current_size, target_size)
            self.ext_data["images"].append(img)

        # Get masks & analyze their properties
        self.logger.info("Calculating masks & properties...")
        masks = [get_sample_mask(img) for img in self.ext_data["images"]]
        mask_regionprops = [
            regionprops_table(mask, properties=("centroid", "feret_diameter_max"))
            for mask in tqdm(masks, desc="Calculating region properties")
        ]
        for d in mask_regionprops:
            for k, v in d.items():
                d[k] = v[0]  # Convert arrays with one entry to scalars

        # Set class attributes:
        self.logger.info("Setting class attributes...")
        self.ext_data["masks"] = masks
        self.ext_df["size"] = sizes
        self.ext_df["context_size"] = context.squeeze()
        self.ext_df.update(mask_regionprops)
        self.ext_df = self.ext_df.astype(float)

        self.logger.info("Extended sources generated.")
        return

    # Function for adding images to the map
    def add_source_image(self, source_arr, coords, centroid=None):
        # Convert coords to pixel coords
        x, y = coords
        x_px, y_px = int((x + 0.5 * self.map_size_deg) * self.map_size_px), int(
            (y + 0.5 * self.map_size_deg) * self.map_size_px
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
        if x_slice.stop > self.map_size_px:
            source_arr = source_arr[: self.map_size_px - x_slice.stop, :]
            x_slice = slice(x_slice.start, self.map_size_px)
        if y_slice.start < 0:
            source_arr = source_arr[:, -y_slice.start :]
            y_slice = slice(0, y_slice.stop)
        if y_slice.stop > self.map_size_px:
            source_arr = source_arr[:, : self.map_size_px - y_slice.stop]
            y_slice = slice(y_slice.start, self.map_size_px)

        # Add source to map
        self.map_array[x_slice, y_slice] += source_arr
        return self.map_array
