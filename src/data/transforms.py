import random
import torch

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
            T.Lambda(single_channel),  # Only one channel
            T.Lambda(minmax_scale),  # Scale to [0, 1]
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.Lambda(random_rotate_90),
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


def random_rotate_90(img):
    return TF.rotate(img, random.choice([0, 90, 180, 270]))
