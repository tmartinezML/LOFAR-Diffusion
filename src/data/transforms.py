import random
import torch
import numpy as np

import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF


def ToTensor():
    transform = T.Compose(
        [
            T.ToImage(),
            T.ToDtype(torch.float32),
        ]
    )
    return transform


def TrainTransform(image_size):
    transform = T.Compose(
        [
            T.CenterCrop(image_size),
            T.Lambda(single_channel),  # Exactly one channel
            T.Lambda(minmax_scale),  # Scale to [0, 1]
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.Lambda(random_rotate_90),
            T.Lambda(train_scale),  # Scale to [-1, 1]
        ]
    )
    return transform


def TrainTransformPrototypes(image_size):
    transform = T.Compose(
        [
            T.CenterCrop(image_size),
            T.Lambda(single_channel),  # Exactly one channel
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(180, interpolation=TF.InterpolationMode.BILINEAR),
            T.Lambda(train_scale),  # Scale to [-1, 1]
        ]
    )
    return transform


def EvalTransform(image_size):
    transform = T.Compose(
        [
            T.Lambda(single_channel),  # Only one channel
            T.Lambda(minmax_scale),  # Scale to [0, 1]
            T.CenterCrop(image_size),
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



def train_scale_present(transform):
    # Check if minmax_scale is part of the composed transform.
    # Used for automatically setting vmin in plotting function.
    return any(
        isinstance(t, T.Lambda) and t.lambd == train_scale for t in transform.transforms
    )


def minmax_scale_batch(batch):
    mx = batch.amax(dim=(-1, -2), keepdim=True)
    mn = batch.amin(dim=(-1, -2), keepdim=True)
    return (batch - mn) / (mx - mn)


def max_scale_batch(batch):
    match batch:
        case torch.Tensor():
            return batch / batch.amax(dim=(-1, -2), keepdim=True)
        case np.ndarray():
            return batch / batch.max(axis=(-1, -2), keepdims=True)
        case _:
            raise TypeError(f"Unsupported type: {type(batch)}")


def random_rotate_90(img):
    return TF.rotate(img, random.choice([0, 90, 180, 270]))
