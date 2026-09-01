import torch
import csv

from datetime import datetime
from tqdm import tqdm

from pathlib import Path

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

EPOCH_LOG_PATH = (
    PROJECT_ROOT / "outputs" / "metrics" / "epoch_metrics.csv"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT / "outputs" / "checkpoints" / "resnet50_best.pt"
)

BATCH_LOG_PATH = (
    PROJECT_ROOT / "outputs" / "metrics" / "batch_metrics.csv"
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

num_epochs = 5
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

    val_loss, val_accuracy, val_macro_f1 = evaluate(
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