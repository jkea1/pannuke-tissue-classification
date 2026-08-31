from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

TISSUE_CLASSES = [
    "Adrenal_gland",
    "Bile-duct",
    "Bladder",
    "Breast",
    "Cervix",
    "Colon",
    "Esophagus",
    "HeadNeck",
    "Kidney",
    "Liver",
    "Lung",
    "Ovarian",
    "Pancreatic",
    "Prostate",
    "Skin",
    "Stomach",
    "Testis",
    "Thyroid",
    "Uterus",
]

Tissue_To_IDX = {
    tissue: idx
    for idx, tissue in enumerate(TISSUE_CLASSES)
}

class PanNukeTissueDataset(Dataset):
    def __init__(self, image_path: str, type_path: str, transform=None):
        self.image_path = Path(image_path)
        self.type_path = Path(type_path)
        self.transform = transform

        self.images = np.load(self.image_path, mmap_mode="r")
        self.types = np.load(self.type_path, mmap_mode="r")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx] # float64 (256, 256, 3), 0~255
        tissue = self.types[idx]
        
        label = Tissue_To_IDX[tissue]

        image = image.astype(np.float32) / 255.0  # -> float32 (256, 256, 3) + HWC + scaling 0~1
        image = np.transpose(image, (2, 0, 1)) # -> float32 (3, 256, 256) CHW
        image = torch.from_numpy(image) # NumPy array -> pyTorch tensor

        if self.transform is not None:
            image = self.transform(image)

        return image, label