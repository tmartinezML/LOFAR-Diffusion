import copy
import random
import logging
from pathlib import Path
from collections.abc import Iterable

import h5py
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import PowerTransformer
from torchvision.transforms.v2 import CenterCrop

import utils.logging
import utils.paths as paths
from data.firstgalaxydata import FIRSTGalaxyData
from plotting.image_plots import plot_image_grid
import data.transforms as T

# Assuming this is in datasets.datasets or a similar module
logger = utils.logging.get_logger(__name__)

def parse_dset_path(dset):
    match dset:
        case Path():
            return dset

        case str():
            if dset in paths.LOFAR_SUBSETS:
                return paths.LOFAR_SUBSETS[dset]

            elif Path(dset).exists():
                return Path(dset)

            else:
                raise FileNotFoundError(f"File {dset} not found.")

        case _:
            raise ValueError(f"Invalid argument type: {dset}")


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dset,
        mode_transforms={"train": T.ToTensor(), "eval": T.ToTensor()},
        train_mode=True,
        selection=None,
        n_subset=None,
        labels=None,
        key="images",
        load_catalog=False,
        catalog_keys=[],
        attributes="all",
        sorted=False,
    ):
        # Set the path for the dataset
        self.path = parse_dset_path(dset)

        self.mode_transforms = mode_transforms
        self.transforms = (
            mode_transforms["train"] if train_mode else mode_transforms["eval"]
        )
        self._context = []

        # Load images
        # If the path is a directory, load all png images in the directory
        if self.path.is_dir():
            self.load_images_png(n_subset)

        # If the path is a hdf5 file, load the images from the file
        elif self.path.suffix in [".hdf5", ".h5"]:
            self.load_images_h5py(
                n_subset,
                selection=selection,
                key=key,
                labels=labels,
                load_catalog=load_catalog,
                catalog_keys=catalog_keys,
                attributes=attributes,
            )

        # If the path is a .pt file, load the images from the file
        elif self.path.suffix == ".pt":
            self.load_images_pt(n_subset)

        # Anything else is not supported.
        else:
            raise ValueError(f"Unknown file type: {self.path.suffix}")

        # Set max values
        if not hasattr(self, "max_values"):
            self.set_max_values()

        # Sort data set by names
        if sorted:
            logger.info("Sorting data set by names...")
            self.sort_by_names()

        # Initialize dict for box-cox transformers
        self.data_transforms = {}

        logger.info("Data set initialized.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]

        if isinstance(i, str):
            i = np.where(self.names == i)[0][0]

        img = self.data[i]
        if self.transforms is not None:
            img = self.transforms(img)
        context = [getattr(self, attr)[i] for attr in self._context]

        if len(context):
            return img, torch.tensor(context)

        else:
            return img

    def set_train_mode(self, train_mode):
        self.transforms = (
            self.mode_transforms["train"]
            if train_mode
            else self.mode_transforms["eval"]
        )

    def index_slice(self, idx):
        # Slice all attributes that have the same shape as self.data
        for attr in self.__dict__.keys():
            if (
                hasattr(self, attr)
                and attr != "data"
                and isinstance(a := getattr(self, attr), Iterable)
                and len(a) == len(self.data)
            ):
                if isinstance(a, pd.DataFrame) and idx.dtype != bool:
                    setattr(self, attr, a.iloc[idx])

                else:
                    setattr(self, attr, a[idx])

        self.data = self.data[idx]

    def index_sliced(self, idx):
        subset = copy.deepcopy(self)
        subset.index_slice(idx)
        return subset

    def sort_by_names(self):
        idxs = np.argsort(self.names)
        self.index_slice(idxs)

    def set_context(self, *args):
        assert all(hasattr(self, attr) for attr in args), (
            "Context attributes not found in dataset: "
            f"{[attr for attr in args if not hasattr(self, attr)]}"
        )
        assert all(len(getattr(self, attr)) == len(self.data) for attr in args), (
            f"Context attributes do not have the same length as data: ({len(self.data)})"
            f"{[(attr, len(getattr(self, attr))) for attr in args if len(getattr(self, attr)) != len(self.data)]}"
        )
        self._context = args

    def load_images_png(self, n_subset=None):
        # Load file names
        files = sorted(self.path.glob("*.png"))

        # Select random subset if desired
        if n_subset is not None:
            logger.info(
                f"Selecting {n_subset} random images" f" from {len(files)} files."
            )
            self.files = random.sample(files, k=n_subset)

        logger.info("Loading images...")

        def load(f):
            return T.ToTensor()(Image.open(f).convert("RGB"))

        self.data = list(map(load, tqdm(files, ncols=80)))
        self.names = [f.stem for f in files]

    def load_images_h5py(
        self,
        n_subset=None,
        selection=None,
        key="images",
        labels=None,
        load_catalog=False,
        catalog_keys=[],
        attributes="all",
    ):

        with h5py.File(self.path, "r") as f:
            images = f[key]
            idxs = slice(None)

            # Load selection if available
            if selection is not None:
                if selection not in f["selections"]:
                    raise ValueError(f"Selection '{selection}' not found in hdf5 file.")
                idxs = f[f"selections/{selection}"][:]

                logger.info(f"Setting selection '{selection}'.")

                # Order might be relevant, but idxs must be in increasing order
                sort_selection = False
                if (np.diff(idxs) < 0).any():
                    logger.info("Selection will be sorted.")
                    sort_selection = True
                    sort_idxs = np.argsort(idxs)
                    idxs = idxs[sort_idxs]

            # Select images with labels if labels are passed
            if labels is not None:
                logger.info(f"Selecting images with label(s) {labels}")
                idxs = np.isin(f[f"{key}_labels"][idxs], labels)

            # Select random subset if n_subset is passed
            n_tot = len(idxs) if hasattr(idxs, "__len__") else images.shape[0]
            if n_subset is not None:
                assert (
                    n_subset <= n_tot
                ), "Requested subset size is larger than total number of selected images."
                logger.info(
                    f"Selecting {n_subset} random images"
                    f" from {n_tot} selected images in hdf5 file."
                )
                idxs = np.sort(np.random.choice(idxs, n_subset, replace=False))

            logger.info("Loading images...")
            self.data = torch.tensor(images[idxs], dtype=torch.float32)

            # See if names are available
            if "names" in f:
                logger.info("Loading names...")
                self.names = np.array(f["names"].asstr()[idxs])
                catalog = None
            elif "catalog" in f.keys():
                logger.info("Loading names from catalog...")
                catalog = pd.read_hdf(self.path, key="catalog")
                self.names = catalog["Source_Name"].values[idxs]

            # Add variable attributes depending on keys in file and specified
            # attributes
            match attributes:
                case "all":
                    attributes = [
                        k
                        for k in f.keys()
                        if k
                        not in [
                            key,  # Usually: images
                            "names",
                            "catalog",
                            "selections",
                            "mask_metadata",
                        ]
                    ]
                case None:
                    attributes = []
                case list():
                    pass
                case _:
                    raise ValueError(f"Invalid value for 'attributes': {attributes}")
            for key in attributes:
                logger.info(f"Loading '{key}'...")
                try:
                    setattr(self, key, torch.tensor(f[key][idxs], dtype=torch.float32))
                except IndexError:
                    logger.error(f"Index error loading '{key}' - will skip.")

            # Load selected attributes if catalog is available
            if len(catalog_keys) or load_catalog:
                if "catalog" in f.keys():
                    logger.info("Loading catalog and entries...")
                    catalog = (
                        catalog
                        if catalog is not None
                        else pd.read_hdf(self.path, key="catalog")
                    )

                    if load_catalog:
                        self.catalog = catalog.iloc[idxs]

                    for key in catalog_keys:
                        values = catalog[key].values[idxs]
                        setattr(
                            self,
                            key,
                            torch.tensor(values) if values.dtype != object else values,
                        )
                else:
                    logger.error("No catalog found in hdf5 file.")

            if not hasattr(self, "names"):
                logger.info("No names loaded from hdf5 file.")
                self.names = np.arange(n_tot)[idxs]

            if selection is not None and sort_selection:
                self.index_slice(sort_idxs)

    def save_selection(self, idxs, name, override=False):

        # Check if file is hdf5
        if self.path.suffix not in [".hdf5", ".h5"]:
            raise ValueError("Saving selection only supported for hdf5 files.")

        # Open file
        with h5py.File(self.path, "r+") as f:

            # Create selecitons group if it does not exist
            if not "selections" in f:
                f.create_group("selections")

            # Get selections group
            grp = f["selections"]

            # Check if selection already exists
            if name in grp:
                if not override:
                    raise ValueError(f"Selection with name '{name}' already exists.")
                else:
                    logger.info(f"Overwriting selection '{name}'.")
                    del grp[name]

            # Save selection
            logger.info(f"Saving selection '{name}'...")
            grp.create_dataset(name, data=idxs)

    def load_images_pt(self, n_subset=None):
        batch_st = torch.load(self.path, map_location="cpu")
        match batch_st.dim():
            # Sampling steps contained:
            case 5:
                samples_itr = torch.clamp(batch_st[:, -1, :, :, :], 0, 1)

            # No sampling steps:
            case 4:
                samples_itr = torch.clamp(batch_st, 0, 1)

            case _:
                raise NotImplementedError(
                    f"Sample batch of unknown shape {batch_st.shape} with {batch_st.dim()} dimensions."
                )
        n_tot = len(samples_itr)

        if n_subset is not None:
            logger.info(
                f"Selecting {n_subset} random images" f" from {len(samples_itr)} files."
            )
            idxs = sorted(random.sample(range(n_tot), k=n_subset))
        else:
            idxs = slice(None)

        self.data = samples_itr[idxs]
        self.names = np.arange(n_tot)[idxs]

    def plot_image_grid(
        self, idxs=None, n_imgs=64, plot_masks=True, show_titles=True, **kwargs
    ):
        # pick n_imgs random images
        idxs = (
            idxs
            if idxs is not None
            else np.random.choice(len(self), n_imgs, replace=False)
        )
        masks = self.masks[idxs] if plot_masks and hasattr(self, "masks") else None
        titles = [f"{self.names[i]}\n({i})" for i in idxs] if show_titles else None
        vmin = -1 if T.train_scale_present(self.transforms) else 0

        # Plot
        return plot_image_grid(
            [self.transforms(self.data[i]) for i in idxs],
            titles=titles,
            masks=masks,
            vmin=vmin,
            **kwargs,
        )

    def set_max_values(self):
        self.max_values = torch.amax(
            self.data,
            dim=(
                -1,
                -2,
            ),
        )

    def transform_max_vals(self):
        # Backwards compatibility
        if not hasattr(self, "max_values"):
            self.set_max_values()

        pt = PowerTransformer(method="box-cox")
        pt.fit(self.max_values.view(-1, 1))
        max_values_tr = pt.transform(self.max_values.view(-1, 1))

        self.max_values_tr = max_values_tr.reshape(self.max_values.shape)
        self.data_transforms["max_values"] = pt
        logger.info(
            f"Max values transformed with Box-Cox transformation ({pt.lambdas_})."
        )

    def box_cox_transform(self, attr):
        if not hasattr(self, attr):
            logger.error(f"Attribute '{attr}' not found.")
            return

        pt = PowerTransformer(method="box-cox")
        pt.fit(getattr(self, attr).view(-1, 1))
        values_tr = pt.transform(getattr(self, attr).view(-1, 1))

        setattr(self, f"{attr}_tr", values_tr.reshape(getattr(self, attr).shape))
        self.data_transforms[attr] = pt
        logger.info(
            f"Attribute '{attr}' transformed with Box-Cox transformation ({pt.lambdas_})."
        )


class LOFARDataset(ImagePathDataset):
    def __init__(self, path, img_size=80, **kwargs):
        super().__init__(
            path,
            mode_transforms={
                "train": T.TrainTransform(img_size),
                "eval": T.EvalTransform(img_size),
            },
            **kwargs,
        )


class SamplesDataset(LOFARDataset):
    def __init__(self, path_or_name, img_size=80, train_mode=False, **kwargs):

        match path_or_name:

            # Samples file
            case Path():
                path = path_or_name

            # Model name
            case str():
                path = (
                    paths.ANALYSIS_PARENT / f"{path_or_name}/{path_or_name}_samples.h5"
                )
                if not path.exists():
                    raise FileNotFoundError(f"File {path} not found.")

            # Invalid argument type
            case _:
                raise ValueError(f"Invalid argument type: {path_or_name}")

        super().__init__(
            path,
            img_size=img_size,
            key="samples",
            train_mode=train_mode,
            **kwargs,
        )
        self.sampling_steps = self.data.numpy().copy()
        self.data = self.data[:, -1, :, :]


class LOFARPrototypesDataset(ImagePathDataset):
    def __init__(self, path, img_size=100, attributes=["masks"], **kwargs):
        super().__init__(
            path,
            mode_transforms={
                "train": T.TrainTransformPrototypes(img_size),
                "eval": T.EvalTransform(img_size),
            },
            attributes=attributes,
            **kwargs,
        )

        # Load mask metadata
        logger.info("Loading mask metadata...")
        self.mask_metadata = pd.read_hdf(self.path, key="mask_metadata")

        # Filter out sources with radius > 0.5 * img_size
        logger.info(
            f"Image size {img_size}: Removing sources with model_radius > {0.5 * img_size}..."
        )
        filter_flag = (self.mask_metadata["Model_Radius"] <= 0.5 * img_size).values
        self.index_slice(filter_flag)
        logger.info(
            f"Removed {(s := (~filter_flag).sum()):_} of {(l := len(filter_flag)):_} sources ({s/l*100:.1f}%)."
        )

        # Center crop images
        logger.info("Reshaping images...")
        self.data = CenterCrop(img_size)(self.data)

        # Bring masks to the same shape as images
        if hasattr(self, "masks"):
            logger.info("Reshaping masks...")
            self.masks = CenterCrop(img_size)(self.masks)

        # Box-Cox transform mask sizes
        self.mask_sizes = torch.Tensor(self.mask_metadata["feret_diameter_max"].values)
        self.box_cox_transform("mask_sizes")


class TrainDatasetFIRST(FIRSTGalaxyData):
    def __init__(self, img_size=80, **kwargs):
        super().__init__(
            selected_split=["train", "test", "valid"],
            is_balanced=True,
            transform=T.TrainTransform(img_size),
            **kwargs,
        )

    def set_context(self, *args):
        logger.warning("FIRSTGalaxyData has class labels as fixed context.")
        return


class CutoutsDataset(LOFARDataset):
    def __init__(self, path, img_size=80, **kwargs):
        super().__init__(path, img_size=img_size, key="cutouts", **kwargs)

        # Add catalog
        logger.info("Adding catalog...")
        cat = pd.read_hdf(path, key="catalog")
        self.catalog = cat

        # Remove problem sources: Flagged by all columns that contain 'Problem'
        problem_flag = cat["Problem_cutout"]
        # Only attributes, not image data, will contain problematic sources.
        for attr in self.__dict__.keys():
            if (
                hasattr(self, attr)
                and attr not in ["data", "catalog"]
                and isinstance(a := getattr(self, attr), Iterable)
                and len(a) == len(cat)
            ):
                setattr(self, attr, getattr(self, attr)[~problem_flag])
        self.catalog = cat[~problem_flag].reset_index(drop=True)
