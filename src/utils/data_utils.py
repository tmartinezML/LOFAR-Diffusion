from pathlib import Path
import os
import random

import torch
import h5py
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, CenterCrop, Lambda
from astropy.stats import sigma_clipped_stats
from tqdm import tqdm
from PIL import Image
from sklearn.preprocessing import PowerTransformer

from utils import paths


def load_data(dataset, batch_size, shuffle=True):
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4,
        drop_last=True,
        pin_memory=True,
    )
    while True:
        yield from loader


def single_channel(img):

    if len(img.shape) == 3:
        return img[:1, :, :]

    elif len(img.shape) == 2:
        return img.unsqueeze(0) if type(img) == torch.Tensor else img[None, :, :]


def train_scale(img):
    return img * 2 - 1


def minmax_scale(img):
    if img.max() == img.min():
        return torch.zeros_like(img)

    return (img - img.min()) / (img.max() - img.min())


def train_transform(image_size):
    transform = Compose([
        # ToTensor(),
        CenterCrop(image_size),
        Lambda(single_channel),  # Only one channel
        Lambda(minmax_scale),  # Scale to [0, 1]
        Lambda(train_scale),  # Scale to [-1, 1]
    ])
    return transform


def eval_transform(image_size):
    transform = Compose([
        Lambda(single_channel),  # Only one channel
        Lambda(minmax_scale),  # Scale to [0, 1]
        CenterCrop(image_size),
    ])
    return transform


def eval_transform_FIRST(image_size):
    transform = Compose([
        ToTensor(),
        Lambda(single_channel),  # Only one channel
        Lambda(minmax_scale),  # Scale to [0, 1]
        CenterCrop(image_size),
    ])
    return transform


def clip_and_rescale(img):
    _, _, stddev = sigma_clipped_stats(data=img.squeeze(),
                                       sigma=3.0, maxiters=10)
    img_clip = torch.clamp(img, 3 * stddev, torch.inf)
    img_norm = (
        (img_clip - torch.min(img_clip))
        / (torch.max(img_clip) - torch.min(img_clip))
    )
    return img_norm


def make_subset(dataset, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    imgs = dataset.data
    names = dataset.filenames
    for img, name in tqdm(zip(imgs, names), total=len(imgs)):
        img.save(f'{out_dir}/{name}.png')


class ImagePathDataset(torch.utils.data.Dataset):
    # From:
    #  https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
    def __init__(
            self, path,
            transforms=ToTensor(),
            n_subset=None,
            labels=None,
            key='images'):

        self.path = path
        self.transforms = transforms
        self._context = []

        # Load images
        if path.is_dir():
            self.load_images_png(n_subset)

        elif path.suffix in [".hdf5", ".h5"]:
            self.load_images_h5py(n_subset, key=key, labels=labels)

        elif path.suffix == ".pt":
            self.load_images_pt(n_subset)

        else:
            raise ValueError(f"Unknown file type: {path.suffix}")

        print("Data set initialized.")

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
            print(
                f"Selecting {n_subset} random images"
                f" from {len(files)} files."
            )
            self.files = random.sample(files, k=n_subset)

        print("Loading images...")
        def load(f): return ToTensor()(Image.open(f).convert("RGB"))
        self.data = list(map(load, tqdm(files, ncols=80)))
        self.names = [f.stem for f in files]

    def load_images_h5py(self, n_subset=None, key='images', labels=None):

        with h5py.File(self.path, 'r') as f:
            images = f[key]

            if labels is not None:
                print(f"Selecting images with label(s) {labels}")
                idxs = np.isin(f[f'{key}_labels'], labels)
                images = images[idxs]

            n_tot = len(images)
            if n_subset is not None:
                assert n_subset <= n_tot, (
                    "Requested subset size is larger than total number of images."
                )
                print(
                    f"Selecting {n_subset} random images"
                    f" from {n_tot} images in hdf5 file."
                )
                idxs = sorted(
                    random.sample(range(n_tot), k=n_subset)
                )
            else:
                idxs = slice(None)
            print("Loading images...")
            self.data = torch.tensor(images[idxs])

            # See if names are available
            if 'names' in f:
                self.names = np.array(f['names'].asstr()[idxs])
            else:
                print("No names loaded from hdf5 file.")
                self.names = np.arange(n_tot)[idxs]

            # Add variable attributes depending on keys in file
            for key in f.keys():
                if key not in ['images', 'names']:
                    setattr(self, key, torch.tensor(f[key][idxs]))

    def load_images_pt(self, n_subset=None):
        batch_st = torch.load(self.path, map_location='cpu')
        samples_itr = torch.clamp(batch_st[:, -1, :, :, :], 0, 1)
        n_tot = len(samples_itr)

        if n_subset is not None:
            print(
                f"Selecting {n_subset} random images"
                f" from {len(samples_itr)} files."
            )
            idxs = sorted(
                random.sample(range(n_tot), k=n_subset)
            )
        else:
            idxs = slice(None)

        self.data = samples_itr[idxs]
        self.names = np.arange(n_tot)[idxs]


class EvaluationDataset(ImagePathDataset):
    def __init__(self, path, img_size=80, **kwargs):
        super().__init__(path, transforms=eval_transform(img_size), **kwargs)


class TrainDataset(ImagePathDataset):
    def __init__(self, path, img_size=80, **kwargs):
        super().__init__(path, transforms=train_transform(img_size), **kwargs)

    def transform_max_vals(self):
        assert hasattr(self, 'max_values'), (
            "Dataset does not contain max values."
        )
        pt = PowerTransformer(method='box-cox')
        pt.fit(self.max_values.view(-1, 1))
        max_values_tr = pt.transform(self.max_values.view(-1, 1))

        self.max_values_tr = max_values_tr.reshape(self.max_values.shape)
        self.box_cox_lambda = pt.lambdas_
        print(
            f"Max values transformed with Box-Cox transformation ({pt.lambdas_}).")
