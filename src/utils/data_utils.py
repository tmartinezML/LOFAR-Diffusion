from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, CenterCrop, Lambda
from astropy.stats import sigma_clipped_stats
from tqdm import tqdm
from PIL import Image

def load_data(dataset, batch_size, shuffle=True):
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1, 
        drop_last=True
    )
    while True:
        yield from loader

def single_channel(img):
    return img[:1,:,:]

def scale(img):
    return img * 2 - 1

def lofar_transform(image_size):
    transform = Compose([
        ToTensor(),
        CenterCrop(image_size),
        Lambda(single_channel),  # Only one channel
        Lambda(scale),  # Scale to [-1, 1]
    ])
    return transform

def clip_and_rescale(img):
    _, _, stddev = sigma_clipped_stats(data=img.squeeze(),
                                       sigma=3.0, maxiters=10)
    img_clip = torch.clamp(img, 3*stddev, torch.inf)
    img_norm = (
        (img_clip - torch.min(img_clip))
        / (torch.max(img_clip) - torch.min(img_clip))
    )
    return img_norm

class ImagePathDataset(torch.utils.data.Dataset):
    # From:
    #  https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
    def __init__(self, path, transforms=ToTensor()):
        self.path = path
        self.files = sorted(self.path.glob("*.png"))
        self.transforms = transforms

        print("Loading images...")
        load = lambda f: Image.open(f).convert("RGB")
        self.data = list(map(load, tqdm(self.files, ncols=80)))

        print("Data set initialized.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = self.data[i]
        if self.transforms is not None:
            img = self.transforms(img)
        return img
    
class LofarSubset(ImagePathDataset):
    image_path = Path("/storage/tmartinez/image_data/lofar_subset")
    def __init__(self, img_size=80):
        super().__init__(self.image_path, transforms=lofar_transform(img_size))

#----------------------------------
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
        raise Exception("Definition: {} is not implemented.".format(definition))


def get_class_dict_rev(definition=global_definition_lit):
    """
    Returns the reverse class definition for the galaxy images.
    :param definition: str, optional
    :return: dict
    """
    class_dict = get_class_dict(definition)
    class_dict_rev = {v: k for k, v in class_dict.items()}
    return class_dict_rev

#----------------------------------
