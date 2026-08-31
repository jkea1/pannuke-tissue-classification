from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.pannuke_tissue.data import PanNukeTissueDataset
from src.pannuke_tissue.models import build_resnet50
from src.pannuke_tissue.train import train_one_epoch
from src.pannuke_tissue.evaluate import evaluate

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PANNUKE_DIR = Path.home() / "datasets" / "PanNuke"

TRAIN_IMAGE_PATH = (
    PANNUKE_DIR / "fold_1" / "images" / "fold1" / "images.npy"
)

TRAIN_TYPE_PATH = (
    PANNUKE_DIR / "fold_1" / "images" / "fold1" / "types.npy"
)

VAL_IMAGE_PATH = (
    PANNUKE_DIR / "fold_2" / "images" / "fold2" / "images.npy"
)

VAL_TYPE_PATH = (
    PANNUKE_DIR / "fold_2" / "images" / "fold2" / "types.npy"
)

transform = transforms.Compose([
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
])

train_dataset = PanNukeTissueDataset(
    TRAIN_IMAGE_PATH,
    TRAIN_TYPE_PATH,
    transform=transform,
)

val_dataset = PanNukeTissueDataset(
    VAL_IMAGE_PATH,
    VAL_TYPE_PATH,
    transform=transform,
)

train_loader = DataLoader(
    train_dataset,
    batch_size = 32,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = build_resnet50().to(device)

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
)

train_loss = train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
)

val_loss = evaluate(
    model,
    val_loader,
    criterion,
    device,
)

print(f"Train loss: {train_loss:.4f}")
print(f"Validation loss: {val_loss:.4f}")