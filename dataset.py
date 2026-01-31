import os
import json
import random
import glob
import torch
from torch.utils.data import Dataset
import numpy as np

from PIL import Image
import PIL.Image
try:
    import pyspng
except ImportError:
    pyspng = None

from datasets import load_from_disk
from torchvision.datasets import ImageFolder
from torchvision import transforms


def center_crop_arr(image_arr, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    pil_image = Image.fromarray(image_arr)
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


class HFImageDataset(Dataset):
    """HuggingFace-based image dataset for iREPA data format."""
    def __init__(self, data_dir, split="train"):
        self.data_dir = data_dir
        split_str = "val" if split == "val" else ""
        self.img_dataset = load_from_disk(os.path.join(data_dir, f"imagenet-latents-images", split_str))

    def __getitem__(self, idx):
        img_elem = self.img_dataset[idx]
        image, label = img_elem["image"], img_elem["label"]
        image = np.array(image.convert("RGB")).transpose(2, 0, 1)
        return torch.from_numpy(image), torch.tensor(label)

    def __len__(self):
        return len(self.img_dataset)


class HFImgLatentDataset(Dataset):
    """HuggingFace-based dataset with both images and precomputed latents."""
    PRECOMPUTED = [
        "sdvae-ft-mse-f8d4",
    ]

    def __init__(self, vae_name, data_dir, split="train"):
        assert vae_name in self.PRECOMPUTED, f"VAE {vae_name} not found in {self.PRECOMPUTED}"
        split_str = "val" if split == "val" else ""
        self.img_dataset = load_from_disk(os.path.join(data_dir, "imagenet-latents-images", split_str))
        self.latent_dataset = load_from_disk(os.path.join(data_dir, f"imagenet-latents-{vae_name}", split_str))
        assert len(self.img_dataset) == len(self.latent_dataset), "Image and latent dataset must have the same length"

    def __getitem__(self, idx):
        img_elem = self.img_dataset[idx]
        image, label = img_elem["image"], img_elem["label"]
        image = np.array(image.convert("RGB")).transpose(2, 0, 1)
        latent = self.latent_dataset[idx]["data"]
        return torch.from_numpy(image), torch.tensor(latent), torch.tensor(label)

    def __len__(self):
        return len(self.img_dataset)


class ImageFolderLatentDataset(Dataset):
    """Dataset using ImageFolder for images and HuggingFace for latents."""
    PRECOMPUTED = [
        "sdvae-ft-mse-f8d4",
    ]

    def __init__(self, vae_name, data_dir, resolution=256, split="train"):
        assert vae_name in self.PRECOMPUTED, f"VAE {vae_name} not found in {self.PRECOMPUTED}"
        vae_split = "val" if split == "val" else ""
        self.img_dataset = ImageFolder(os.path.join(data_dir, "imagenet", split))
        self.transform_train = transforms.Lambda(
            lambda img: center_crop_arr(np.array(img.convert("RGB")), resolution)
        )
        self.latent_dataset = load_from_disk(os.path.join(data_dir, f"imagenet-latents-{vae_name}", vae_split))
        assert len(self.img_dataset) == len(self.latent_dataset), "Image and latent dataset must have the same length"

    def __getitem__(self, idx):
        image, label = self.img_dataset[idx]
        image = self.transform_train(image)
        image = image.transpose(2, 0, 1)
        latent = self.latent_dataset[idx]["data"]
        return torch.from_numpy(image), torch.tensor(latent), torch.tensor(label)

    def __len__(self):
        return len(self.img_dataset)


class HFLatentDataset(Dataset):
    """HuggingFace-based dataset with only latents (no images)."""
    PRECOMPUTED = [
        "sdvae-ft-mse-f8d4",
    ]

    def __init__(self, vae_name, data_dir, split="train"):
        split_str = "val" if split == "val" else ""
        assert vae_name in self.PRECOMPUTED, f"VAE {vae_name} not found in {self.PRECOMPUTED}"
        assert os.path.exists(os.path.join(data_dir, f"imagenet_{split}_labels.txt")), \
            "imagenet_train_labels.txt not found, please download from huggingface"

        self.latent_dataset = load_from_disk(os.path.join(data_dir, f"imagenet-latents-{vae_name}", split_str))

        with open(os.path.join(data_dir, f"imagenet_{split}_labels.txt"), "r") as f:
            self.labels = [int(line.strip()) for line in f.readlines()]

    def __getitem__(self, idx):
        latent = self.latent_dataset[idx]["data"]
        label = self.labels[idx]
        return torch.tensor(latent), torch.tensor(label)

    def __len__(self):
        return len(self.latent_dataset)

