from pathlib import Path
import csv

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.pannuke_tissue.data import PanNukeTissueDataset, TISSUE_CLASSES
from src.pannuke_tissue.models import build_resnet50
from src.pannuke_tissue.evaluate import evaluate

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PANNUKE_DIR = Path.home() / "datasets" / "PanNuke"

VAL_IMAGE_PATH = (
    PANNUKE_DIR / "fold_2" / "images" / "fold2" / "images.npy"
)

VAL_TYPE_PATH = (
    PANNUKE_DIR / "fold_2" / "images" / "fold2" / "types.npy"
)

EXPERIMENT_NAME = "06_resnet50_classweighted_50ep_lr5e-5_bs32_seed42"

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / f"{EXPERIMENT_NAME}_best.pt"
)

OUTPUT_CSV_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / f"{EXPERIMENT_NAME}_per_class_metrics.csv"
)


BATCH_SIZE = 32

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


val_transform = transforms.Compose([
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

val_dataset = PanNukeTissueDataset(
    VAL_IMAGE_PATH,
    VAL_TYPE_PATH,
    transform=val_transform,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


model = build_resnet50().to(device)

state_dict = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
)

model.load_state_dict(state_dict)


criterion = torch.nn.CrossEntropyLoss()

val_loss, val_accuracy, val_macro_f1, per_class_metrics = evaluate(
    model,
    val_loader,
    criterion,
    device,
)


print(f"Val loss: {val_loss:.4f}")
print(f"Val accuracy: {val_accuracy:.4f}")
print(f"Val macro-F1: {val_macro_f1:.4f}")


with open(OUTPUT_CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "class_name",
        "precision",
        "recall",
        "f1_score",
        "support",
    ])

    for class_name in TISSUE_CLASSES:
        metrics = per_class_metrics[class_name]

        writer.writerow([
            class_name,
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            metrics["support"],
        ])


print(f"Saved: {OUTPUT_CSV_PATH}")