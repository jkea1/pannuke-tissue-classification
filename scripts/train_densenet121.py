import random
import numpy as np
import torch
import csv

from datetime import datetime
from tqdm import tqdm

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms

from src.pannuke_tissue.data import PanNukeTissueDataset
from src.pannuke_tissue.models import build_densenet121
from src.pannuke_tissue.train import train_one_epoch
from src.pannuke_tissue.evaluate import evaluate

EXPERIMENT_NAME = "07_densenet121_augmentation_50ep_lr5e-5_bs32_seed42"

SEED = 42
NUM_WORKERS = 0
NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 5e-5

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

g = torch.Generator()
g.manual_seed(SEED)

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

EPOCH_LOG_PATH = (
    PROJECT_ROOT / "outputs" / "metrics" / f"{EXPERIMENT_NAME}_epoch_metrics.csv"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT / "outputs" / "checkpoints" / f"{EXPERIMENT_NAME}_best.pt"
)

BATCH_LOG_PATH = (
    PROJECT_ROOT / "outputs" / "metrics" / f"{EXPERIMENT_NAME}_batch_metrics.csv"
)

BATCH_LOG_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

EPOCH_LOG_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with open(BATCH_LOG_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "timestamp",
        "epoch",
        "batch",
        "train_loss",
    ])

with open(EPOCH_LOG_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "timestamp",
        "epoch",
        "train_loss",
        "val_loss",
        "val_accuracy",
        "val_macro_f1",
        "best_checkpoint",
    ])

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

val_transform = transforms.Compose([
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

train_dataset = PanNukeTissueDataset(
    TRAIN_IMAGE_PATH,
    TRAIN_TYPE_PATH,
    transform=train_transform,
)

val_dataset = PanNukeTissueDataset(
    VAL_IMAGE_PATH,
    VAL_TYPE_PATH,
    transform=val_transform,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    generator=g,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = build_densenet121().to(device)

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

num_epochs = NUM_EPOCHS
best_val_macro_f1 = -1.0
best_epoch = -1

epoch_records = []

overall_bar = tqdm(
    range(1, num_epochs + 1),
    desc="Overall training",
    unit="epoch",
    position=0,

)

for epoch in overall_bar:
    train_loss = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
        epoch=epoch,
        batch_log_path=BATCH_LOG_PATH,
        log_interval=10,
    )

    val_loss, val_accuracy, val_macro_f1, _ = evaluate(
        model,
        val_loader,
        criterion,
        device,
    )

    epoch_records.append([
        datetime.now().isoformat(timespec="seconds"),
        epoch,
        train_loss,
        val_loss,
        val_accuracy,
        val_macro_f1,
    ])

    if val_macro_f1 > best_val_macro_f1:
        best_val_macro_f1 = val_macro_f1
        best_epoch = epoch

        torch.save(
            model.state_dict(),
            CHECKPOINT_PATH,
        )

    overall_bar.set_postfix_str(
        f"train_loss={train_loss:.4f} | "
        f"val_loss={val_loss:.4f} | "
        f"val_macro_f1={val_macro_f1:.4f}"
    )

with open(EPOCH_LOG_PATH, "a", newline="") as f:
    writer = csv.writer(f)

    for record in epoch_records:
        epoch = record[1]

        best_checkpoint = (
            "Best"
            if epoch == best_epoch
            else ""
        )

        writer.writerow([
            *record,
            best_checkpoint,
        ])

print(
    f"Training complete! "
    f"Best epoch: {best_epoch} | "
    f"Best val macro-F1: {best_val_macro_f1:.4f}"
)