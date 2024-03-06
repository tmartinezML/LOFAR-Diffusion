#!/usr/bin/env python
import logging
import shutil
from matplotlib import pyplot as plt
import numpy as np
import os
from torch.utils.data import Dataset

from tqdm import tqdm

from PIL import Image
# import glob
import pandas as pd
import random
from astropy import units as u
from astropy.coordinates import SkyCoord

import torch
from torchvision.datasets.utils import download_url  # , check_integrity
from torchvision.transforms import ToTensor

# Settings for Unlabeled / zoomde-in:
# las: 40 - 120 / 0
# flux: 5 / 0.75
# peak flux: False / True


class LofarUnlabeled(Dataset):
    """
    LofarUnlabeled class provides unlabeled images from LoTSS DR2
    """

    def __init__(self, img_dir, catalog_file,
                 las_thr=0., flux_thr=0., peak_flux=False,
                 datapoints=None, transform=ToTensor(),
                 apply_mask=False, array_format=False, RGB_format=False, _seed=None):

        self.las_thr = las_thr
        self.flux_thr = flux_thr
        self.datapoints = datapoints
        self.transform = transform
        self.apply_mask = apply_mask
        self.array_format = array_format
        self.RGB_format = RGB_format
        self._seed = _seed

        # Import catalog csv file
        annotation = pd.read_csv(catalog_file)

        # Remove sources with problem or nan labels:
        annotation = self.remove_bad_imgs(annotation)

        # Apply cuts:
        # Size threshold
        if self.las_thr != 0.:
            annotation = self.annotation_threshold_cut(
                annotation, self.las_thr, 'LAS'
            )
        # Flux threshold
        if self.flux_thr != 0.:  # Flux threshold
            annotation = self.annotation_threshold_cut(
                annotation, self.flux_thr,
                ('Peak_flux' if peak_flux else 'Total_flux')
            )

        if self.datapoints is not None:  # Limit amount of data
            if self.datapoints < len(annotation):
                np.random.seed(42)  # seed for datasample reproducibility
                self.inds = np.random.choice(
                    len(annotation), self.datapoints, replace=False
                )
                annotation = annotation.iloc[self.inds]

        # Collect data:
        print("Loading images...")

        def load(s):
            with Image.open(os.path.join(img_dir, s) + '.png') as img:
                return img.copy()

        self.data = list(map(load, tqdm(annotation["Source_Name"])))
        print("Collecting annotation data...")
        self.filenames = annotation["Source_Name"].to_list()
        self.las = annotation["LAS"].to_list()
        self.flux = annotation["Total_flux"].to_list()
        self.coordinates = {"RA": annotation["RA"], "DEC": annotation["DEC"]}
        self.mask_params = []
        if self.apply_mask:
            print("Collecting mask data...")
            for _, source in tqdm(annotation.iterrows()):
                # Assigment motivated from LAS parameter: LGZ_Size or 2*DC_Maj
                self.mask_params.append(
                    ((source['LGZ_PA'], source['DC_PA']),
                     (source['LGZ_Size'], 2 * source['DC_Maj']),
                     (source['LGZ_Width'], 2 * source['DC_Min']))
                )
        print("Data set initialized.")

    def annotation_threshold_cut(self, annotation, thr, label):
        if isinstance(self.flux_thr, list):
            mask = annotation[label].between(thr[0], thr[1])
        else:
            mask = annotation[label] > thr

        print(f"Excluding {len(annotation) - np.sum(mask)} sources with "
              f"{label} < {thr}.")
        return annotation[mask]

    def remove_bad_imgs(self, annotation):
        """
        Exclude images with problems in the data reduction process.
        This excludes a total of 456 sources.
        """
        mask = (annotation['Problem_norm'] == False) &\
            (annotation['Problem_resize'] == False) &\
            (annotation['Problem_clip'] == False) &\
            (annotation['Problem_cutout'] == False) &\
            (annotation['Nans_cutout'] == False)

        print(f"Excluding {len(annotation) - np.sum(mask)} sources with "
              f"problems in the data reduction process.")

        annotation = annotation[mask]
        return annotation

    def __getitem__(self, index):
        img = self.data[index]
        if self.apply_mask:
            source_PA = self.mask_params[index][0][0] if not np.isnan(
                self.mask_params[index][0][0]) else self.mask_params[index][0][1]
            if np.isnan(source_PA):
                print(f"PA is NaN for index {index}!")
            source_size = self.mask_params[index][1][0] if not np.isnan(
                self.mask_params[index][1][0]) else self.mask_params[index][1][1]
            if np.isnan(source_PA):
                print(f"PA is NaN for index {index}!")
            source_width = self.mask_params[index][2][0] if not np.isnan(
                self.mask_params[index][2][0]) else self.mask_params[index][2][1]
            if np.isnan(source_PA):
                print(f"PA is NaN for index {index}!")
            img = self.__mask_image__(
                img, source_PA, source_size, source_width)
        if self.array_format:
            img = np.array(img) / 255
        if self.RGB_format:
            img = img.convert("RGB")
        if self.transform is not None:
            if self._seed is not None:
                random.seed(self._seed)
                torch.manual_seed(self._seed)
            img = self.transform(img)
        return img

    def __get_indices__(self):
        return self.inds

    def __mask_image__(self, img, source_PA, source_size, source_width):
        m = np.zeros(img.shape, dtype=np.uint8)
        CDL1_safety_factor = 2
        resolution_factor = 300 / 450 * CDL1_safety_factor  # "px" / "arcsec"
        source_size = resolution_factor * source_size
        source_width = resolution_factor * source_width
        x0 = int(m.shape[0] / 2) - 0.5
        y0 = int(m.shape[1] / 2) - 0.5
        for x in range(m.shape[0]):
            for y in range(m.shape[1]):
                if (np.power((x - x0) / (source_width / 2), 2) + np.power((y - y0) / (source_size / 2), 2)) <= 1:
                    m[x, y] = 1
        m_pil = Image.fromarray(m, mode="L")
        mask = m_pil.rotate(90.0 - source_PA)
        mask = np.array(mask)
        img_masked = np.multiply(img, mask)
        return img_masked

    def __getcoords__(self, index):
        RA = self.coordinates["RA"].iloc[index]
        DEC = self.coordinates["DEC"].iloc(index)
        return SkyCoord(RA, DEC, unit=(u.deg, u.deg))

    def __getmaskparam__(self, index):
        return self.mask_params[index]

    def __gethistogram__(self):
        histogram = [np.sum(item) for item in self.data]
        return histogram

    def __len__(self):
        return len(self.data)

    def convert_img_np(self, img_from_database):
        img_back = np.squeeze(np.asarray(img_from_database))
        img_back_uint8 = img_back.astype(np.uint8)
        return img_back_uint8

    def show_coords(self):
        plt.figure(figsize=(8, 4.2))
        plt.subplot(111, projection="aitoff")
        plt.title("Aitoff projection of coordinates")
        plt.grid(True)
        for c in self.coordinates:
            if c is not None:
                ra_rad = c.ra.wrap_at(180 * u.deg).radian
                dec_rad = c.dec.radian
                plt.plot(ra_rad, dec_rad, 'o', markersize=1.5,
                         color="red", alpha=0.3)

        plt.subplots_adjust(top=0.95, bottom=0.0)
        plt.show()
