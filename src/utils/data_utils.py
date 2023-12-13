from pathlib import Path
import os
import random

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, CenterCrop, Lambda
from astropy.stats import sigma_clipped_stats
from tqdm import tqdm
from PIL import Image


def load_data(dataset, batch_size, shuffle=True):
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4,
        drop_last=True,
        pin_memory=True,
    )
    while True:
        yield from loader


def single_channel(img):
    return img[:1, :, :]


def scale(img):
    return img * 2 - 1


def train_transform(image_size):
    transform = Compose([
        ToTensor(),
        CenterCrop(image_size),
        Lambda(single_channel),  # Only one channel
        Lambda(scale),  # Scale to [-1, 1]
    ])
    return transform


def eval_transform(image_size):
    transform = Compose([
        ToTensor(),
        CenterCrop(image_size),
        Lambda(single_channel),  # Only one channel
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
    def __init__(self, path, transforms=ToTensor(), n_subset=None):
        self.path = path
        self.files = sorted(self.path.glob("*.png"))
        self.transforms = transforms

        if n_subset is not None:
            print(
                f"Selecting {n_subset} random images"
                f" from {len(self.files)} files."
            )
            self.files = random.choices(self.files, k=n_subset)

        print("Loading images...")
        def load(f): return Image.open(f).convert("RGB")
        self.data = list(map(load, tqdm(self.files, ncols=80)))

        print("Data set initialized.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = self.data[i]
        if self.transforms is not None:
            img = self.transforms(img)
        return img


class EvaluationDataset(ImagePathDataset):
    def __init__(self, path, img_size=80, **kwargs):
        super().__init__(path, transforms=eval_transform(img_size), **kwargs)


class LofarSubset(ImagePathDataset):
    image_path = Path("/storage/tmartinez/image_data/lofar_subset")

    def __init__(self, img_size=80, **kwargs):
        super().__init__(self.image_path, transforms=train_transform(img_size),
                         **kwargs)


class LofarDummySet(ImagePathDataset):
    image_path = Path("/home/bbd0953/diffusion/image_data/dummy")

    def __init__(self, img_size=80):
        super().__init__(self.image_path, transforms=train_transform(img_size))


class LofarZoomUnclipped80(ImagePathDataset):
    image_path = Path(
        "/home/bbd0953/diffusion/image_data/lofar_zoom_unclipped_subset_80p"
    )

    def __init__(self, img_size=80, **kwargs):
        super().__init__(self.image_path, transforms=train_transform(img_size),
                          **kwargs)

# ----------------------------------
# FIRST Dataset:


global_definition_lit = "literature"
global_definition_cdl1 = "CDL1"


def get_class_dict(definition=global_definition_lit):
    """
    Returns the class definition for the galaxy images.
    :param definition: str, optional
        either literature or CDL1
    :return: dict
    """
    if definition == global_definition_lit:
        return {0: "FRI",
                1: "FRII",
                2: "Compact",
                3: "Bent"}
    elif definition == global_definition_cdl1:
        return {0: "FRI-Sta",
                1: "FRII",
                2: "Compact",
                3: "FRI-WAT",
                4: "FRI-NAT"}
    else:
        raise Exception(
            "Definition: {} is not implemented.".format(definition))


def get_class_dict_rev(definition=global_definition_lit):
    """
    Returns the reverse class definition for the galaxy images.
    :param definition: str, optional
    :return: dict
    """
    class_dict = get_class_dict(definition)
    class_dict_rev = {v: k for k, v in class_dict.items()}
    return class_dict_rev

# ----------------------------------
