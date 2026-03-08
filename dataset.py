import os
import io
import json
import random
import glob
import zipfile
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


class EDM2ImgLatentDataset(Dataset):
    """Dataset for edm2/REPA-style preprocessed data (ZIP or folder with PNG images + npy latents).

    Expected directory layout under data_dir:
        images/          (or images.zip)  — PNG files + dataset.json with labels
        vae-sd/          (or vae-sd.zip)  — img-mean-std-*.npy files + dataset.json

    Both produced by preprocessing/dataset_tools.py (convert + encode).
    """

    def __init__(self, data_dir, images_name="images", latents_name="vae-sd", split="train"):
        self.data_dir = data_dir

        # Resolve image source (folder or zip)
        img_path = os.path.join(data_dir, images_name)
        img_zip_path = img_path + ".zip"
        if os.path.isdir(img_path):
            self._img_zip = None
            self._img_root = img_path
        elif os.path.isfile(img_zip_path):
            self._img_zip = img_zip_path
            self._img_root = None
        else:
            raise FileNotFoundError(f"Neither {img_path} nor {img_zip_path} found")

        # Resolve latent source (folder or zip)
        lat_path = os.path.join(data_dir, latents_name)
        lat_zip_path = lat_path + ".zip"
        if os.path.isdir(lat_path):
            self._lat_zip = None
            self._lat_root = lat_path
        elif os.path.isfile(lat_zip_path):
            self._lat_zip = lat_zip_path
            self._lat_root = None
        else:
            raise FileNotFoundError(f"Neither {lat_path} nor {lat_zip_path} found")

        # Load image file list and labels
        self._img_files, self._labels = self._load_manifest(
            self._img_root, self._img_zip, ext=".png"
        )
        # Load latent file list
        self._lat_files, _ = self._load_manifest(
            self._lat_root, self._lat_zip, ext=".npy"
        )
        assert len(self._img_files) == len(self._lat_files), (
            f"Image count ({len(self._img_files)}) != latent count ({len(self._lat_files)})"
        )

    @staticmethod
    def _load_manifest(root, zip_path, ext):
        """Load sorted file list and labels from folder or zip."""
        if root is not None:
            meta_path = os.path.join(root, "dataset.json")
            if os.path.isfile(meta_path):
                with open(meta_path, "r") as f:
                    meta = json.load(f)
            else:
                meta = {"labels": None}
            # Collect files
            all_files = []
            for dirpath, _, filenames in os.walk(root):
                for fn in filenames:
                    if fn.lower().endswith(ext):
                        rel = os.path.relpath(os.path.join(dirpath, fn), root)
                        all_files.append(rel)
            all_files.sort()
        else:
            with zipfile.ZipFile(zip_path, "r") as zf:
                meta_data = zf.read("dataset.json") if "dataset.json" in zf.namelist() else b'{"labels": null}'
                meta = json.loads(meta_data)
                all_files = sorted([f for f in zf.namelist() if f.lower().endswith(ext)])

        # Build label dict
        labels_list = meta.get("labels")
        label_dict = {}
        if labels_list is not None:
            label_dict = {item[0]: item[1] for item in labels_list}

        labels = [label_dict.get(f, -1) for f in all_files]
        return all_files, labels

    def _read_image(self, idx):
        fname = self._img_files[idx]
        if self._img_root is not None:
            full = os.path.join(self._img_root, fname)
            img = np.array(Image.open(full).convert("RGB"))
        else:
            with zipfile.ZipFile(self._img_zip, "r") as zf:
                with zf.open(fname) as f:
                    img = np.array(Image.open(f).convert("RGB"))
        return img.transpose(2, 0, 1)  # HWC -> CHW

    def _read_latent(self, idx):
        fname = self._lat_files[idx]
        if self._lat_root is not None:
            full = os.path.join(self._lat_root, fname)
            latent = np.load(full)
        else:
            with zipfile.ZipFile(self._lat_zip, "r") as zf:
                with zf.open(fname) as f:
                    latent = np.load(io.BytesIO(f.read()))
        return latent  # shape [8, H/8, W/8] = [mean(4ch), std(4ch)]

    def __getitem__(self, idx):
        image = self._read_image(idx)
        latent = self._read_latent(idx)
        label = self._labels[idx]
        return torch.from_numpy(image), torch.tensor(latent), torch.tensor(label)

    def __len__(self):
        return len(self._img_files)

    def __repr__(self):
        return f"EDM2ImgLatentDataset(n={len(self)}, img_root={self._img_root or self._img_zip})"

