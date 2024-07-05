import os
import random

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from astropy.stats import sigma_clipped_stats
import torchvision.transforms.v2.functional as TF
from torchvision.transforms.v2 import (
    Lambda,
    Compose,
    ToImage,
    ToDtype,
    CenterCrop,
    RandomVerticalFlip,
    RandomHorizontalFlip,
)


def load_data(dataset, batch_size, shuffle=True):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        drop_last=True,
        pin_memory=True,
    )
    while True:
        yield from loader


def to_tensor():
    transform = Compose(
        [
            ToImage(),
            ToDtype(torch.float32),
        ]
    )
    return transform


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


def random_rotate_90(img):
    return TF.rotate(img, random.choice([0, 90, 180, 270]))


def train_transform(image_size):
    transform = Compose(
        [
            CenterCrop(image_size),
            Lambda(single_channel),  # Only one channel
            Lambda(minmax_scale),  # Scale to [0, 1]
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            Lambda(random_rotate_90),
            Lambda(train_scale),  # Scale to [-1, 1]
        ]
    )
    return transform


def eval_transform(image_size):
    transform = Compose(
        [
            Lambda(single_channel),  # Only one channel
            Lambda(minmax_scale),  # Scale to [0, 1]
            CenterCrop(image_size),
        ]
    )
    return transform


def eval_transform_FIRST(image_size):
    transform = Compose(
        [
            Lambda(single_channel),  # Only one channel
            Lambda(minmax_scale),  # Scale to [0, 1]
            CenterCrop(image_size),
        ]
    )
    return transform


def clip_and_rescale(img):
    _, _, stddev = sigma_clipped_stats(data=img.squeeze(), sigma=3.0, maxiters=10)
    img_clip = torch.clamp(img, 3 * stddev, torch.inf)
    img_norm = (img_clip - torch.min(img_clip)) / (
        torch.max(img_clip) - torch.min(img_clip)
    )
    return img_norm


def make_subset(dataset, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    imgs = dataset.data
    names = dataset.filenames
    for img, name in tqdm(zip(imgs, names), total=len(imgs)):
        img.save(f"{out_dir}/{name}.png")
