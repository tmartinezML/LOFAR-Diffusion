import numpy as np
import pandas as pd
from tqdm import tqdm

import maps.map_utils as mutil
import model.sampler as smplr
from data.segment import get_sample_mask

# TODO:
# - Add type hints
# - Add docstrings
# - Review variable names
# - Add logger


class MapMaker:
    def __init__(self, map_size_px, map_size_deg, model_name, img_size=80,):

        self.map_size_px = map_size_px
        self.map_size_deg = map_size_deg
        self.map_array = np.zeros((map_size_px, map_size_px))
        self.img_size = img_size
        self.model_name = model_name

        # Extended sources
        self.ext_data = {
            "images": None,
            "masks": None,
        }
        self.ext_df = pd.DataFrame(
            columns=[
                "x_coord",
                "y_coord",
                "size",
                "context_size",
                "flux",
                "centroid-0",
                "centroid-1",
                "feret_diameter_max",
            ]
        )

        # Compact sources
        self.comp_images = None
        self.comp_df = pd.DataFrame(
            columns=["x_coord", "y_coord", "size", "flux", "angle", "bmin", "bmaj"]
        )


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
            source_arr = (
                mutil.gaussian_signal(
                    size=source.size, angle=source.angle, convolve=True, img_size=self.img_size
                )
            )
            self.comp_images[i] = source_arr
            source_arr *= source.flux  # Gaussian signal is normalized already

            # Add source array to map
            self.add_source_image(source_arr, coords)


    def add_extended_sources(self):
        # Place AGNs on map such that centroid matches the catalog x, y position
        if self.ext_data["images"] is None:
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

            # Get source array. If necessary, upscale image
            if (target_size := source.size[i]) != (current_size := source.context_size):
                img = mutil.upscale_image(
                    self.ext_data["images"][i], current_size, target_size
                )
                mask = get_sample_mask(img)
                source_arr = img * mask
            else:
                source_arr = self.ext_data["images"][i] * self.ext_data["masks"][i]

            # Scale source to flux
            source_arr = mutil.scale_to_flux(source_arr, source.flux)

            # Add source array to map
            self.add_source_image(source_arr, coords, centroid=centroid)
    
    def make_extended_sources(self):
        sampler = smplr.Sampler(self.model_name)
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
