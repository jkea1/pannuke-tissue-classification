import csv
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.pannuke_tissue.data import (
    PanNukeTissueDataset,
    TISSUE_CLASSES,
)
from src.pannuke_tissue.models import build_swin_t
from src.pannuke_tissue.evaluate import evaluate

# --------------------------------------------------
# Configuration
# --------------------------------------------------

BATCH_SIZE = 32
NUM_WORKERS = 0

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PANNUKE_DIR = Path.home() / "datasets" / "PanNuke"

# Fold 3 = final test set
TEST_IMAGE_PATH = (
    PANNUKE_DIR
    / "fold_3"
    / "images"
    / "fold3"
    / "images.npy"
)

TEST_TYPE_PATH = (
    PANNUKE_DIR
    / "fold_3"
    / "images"
    / "fold3"
    / "types.npy"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "08_swin_t_augmentation_50ep_lr5e-5_bs32_seed42_best.pt"
)

TEST_METRICS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "swin_t_final_test_per_class_metrics.csv"
)

CONFUSION_MATRIX_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "swin_t_final_test_confusion_matrix.png"
)


NORMALIZED_CONFUSION_MATRIX_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "swin_t_final_test_confusion_matrix_normalized.png"
)

CONFUSION_MATRIX_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

NORMALIZED_CONFUSION_MATRIX_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

# --------------------------------------------------
# Test preprocessing
# --------------------------------------------------

test_transform = transforms.Compose([
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# --------------------------------------------------
# Dataset / DataLoader
# --------------------------------------------------

test_dataset = PanNukeTissueDataset(
    TEST_IMAGE_PATH,
    TEST_TYPE_PATH,
    transform=test_transform,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)

# --------------------------------------------------
# Model
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = build_swin_t().to(device)

model.load_state_dict(
    torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )
)

# --------------------------------------------------
# Evaluation
# --------------------------------------------------

criterion = torch.nn.CrossEntropyLoss()

test_loss, test_accuracy, test_macro_f1, per_class_metrics, all_labels, all_predictions = evaluate(
    model,
    test_loader,
    criterion,
    device,
    return_predictions=True,
)

# --------------------------------------------------
# Print overall results
# --------------------------------------------------

print("\nFinal Test Results")
print("------------------")
print(f"Test loss:     {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test Macro-F1: {test_macro_f1:.4f}")

# --------------------------------------------------
# Save per-class metrics
# --------------------------------------------------

TEST_METRICS_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with open(TEST_METRICS_PATH, "w", newline="") as f:
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


# --------------------------------------------------
# Compute confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(len(TISSUE_CLASSES))),
)

# --------------------------------------------------
# Plot confusion matrix
# --------------------------------------------------

fig, ax = plt.subplots(
    figsize=(14, 14)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=TISSUE_CLASSES,
)

disp.plot(
    ax=ax,
    xticks_rotation=90,
    values_format="d",
    cmap="Blues",
)

plt.title(
    "Swin-T Confusion Matrix - Fold 3 Test Set"
)

plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=300,
)

plt.close()

# --------------------------------------------------
# Normalized confusion matrix
# --------------------------------------------------

cm_normalized = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(len(TISSUE_CLASSES))),
    normalize="true",
)

fig, ax = plt.subplots(figsize=(14, 14))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_normalized,
    display_labels=TISSUE_CLASSES,
)

disp.plot(
    ax=ax,
    xticks_rotation=90,
    values_format=".2f",
    cmap="Blues",
)

plt.title(
    "Swin-T Normalized Confusion Matrix - Fold 3 Test Set"
)

plt.tight_layout()

plt.savefig(
    NORMALIZED_CONFUSION_MATRIX_PATH,
    dpi=300,
)

plt.close()

# --------------------------------------------------
# Compare Uterus -> Cervix misclassifications
# with correctly classified Cervix samples
# --------------------------------------------------

UTERUS_IDX = TISSUE_CLASSES.index("Uterus")
CERVIX_IDX = TISSUE_CLASSES.index("Cervix")

COMPARISON_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "uterus_cervix_comparison.png"
)

COMPARISON_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

# True: Uterus, Predicted: Cervix
uterus_as_cervix_indices = [
    idx
    for idx, (true_label, predicted_label) in enumerate(
        zip(all_labels, all_predictions)
    )
    if true_label == UTERUS_IDX
    and predicted_label == CERVIX_IDX
]

# True: Cervix, Predicted: Cervix
correct_cervix_indices = [
    idx
    for idx, (true_label, predicted_label) in enumerate(
        zip(all_labels, all_predictions)
    )
    if true_label == CERVIX_IDX
    and predicted_label == CERVIX_IDX
]

print(
    f"Uterus -> Cervix misclassified samples: "
    f"{len(uterus_as_cervix_indices)}"
)

print(
    f"Correctly classified Cervix samples: "
    f"{len(correct_cervix_indices)}"
)


# --------------------------------------------------
# Helper function to undo normalization
# --------------------------------------------------

def denormalize_image(image):
    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(3, 1, 1)

    image = image * std + mean
    image = image.clamp(0, 1)

    return image.permute(1, 2, 0).numpy()

# --------------------------------------------------
# Plot comparison
# --------------------------------------------------

NUM_EXAMPLES = 5

num_examples = min(
    NUM_EXAMPLES,
    len(uterus_as_cervix_indices),
    len(correct_cervix_indices),
)

fig, axes = plt.subplots(
    2,
    num_examples,
    figsize=(3 * num_examples, 7),
)

for i in range(num_examples):

    # Uterus misclassified as Cervix
    dataset_idx = uterus_as_cervix_indices[i]

    image, _ = test_dataset[dataset_idx]
    image = denormalize_image(image)

    axes[0, i].imshow(image)
    axes[0, i].axis("off")

    # Correctly classified Cervix
    dataset_idx = correct_cervix_indices[i]

    image, _ = test_dataset[dataset_idx]
    image = denormalize_image(image)

    axes[1, i].imshow(image)
    axes[1, i].axis("off")


# Main title
fig.suptitle(
    "Uterus-Cervix Classification Examples",
    fontsize=16,
    y=0.98,
)

# Row titles
axes[0, 0].set_title(
    "True: Uterus | Predicted: Cervix",
    fontsize=12,
    loc="left",
    pad=10,
)

axes[1, 0].set_title(
    "True: Cervix | Predicted: Cervix",
    fontsize=12,
    loc="left",
    pad=10,
)

plt.tight_layout(
    rect=[0, 0, 1, 0.95]
)

# --------------------------------------------------
# Plot comparison
# --------------------------------------------------

NUM_EXAMPLES = 5

num_examples = min(
    NUM_EXAMPLES,
    len(uterus_as_cervix_indices),
    len(correct_cervix_indices),
)

fig, axes = plt.subplots(
    2,
    num_examples,
    figsize=(3 * num_examples, 7),
)

for i in range(num_examples):

    # Uterus misclassified as Cervix
    dataset_idx = uterus_as_cervix_indices[i]

    image, _ = test_dataset[dataset_idx]
    image = denormalize_image(image)

    axes[0, i].imshow(image)
    axes[0, i].axis("off")

    # Correctly classified Cervix
    dataset_idx = correct_cervix_indices[i]

    image, _ = test_dataset[dataset_idx]
    image = denormalize_image(image)

    axes[1, i].imshow(image)
    axes[1, i].axis("off")


# Main title
fig.suptitle(
    "Uterus-Cervix Classification Examples",
    fontsize=16,
    y=0.98,
)

# Row titles
axes[0, 0].set_title(
    "True: Uterus | Predicted: Cervix",
    fontsize=12,
    loc="left",
    pad=10,
)

axes[1, 0].set_title(
    "True: Cervix | Predicted: Cervix",
    fontsize=12,
    loc="left",
    pad=10,
)

plt.tight_layout(
    rect=[0, 0, 1, 0.95]
)

fig.subplots_adjust(
    hspace=0.18,
)

plt.savefig(
    COMPARISON_PATH,
    dpi=300,
)

plt.close()

print(
    f"Uterus-Cervix comparison saved to:\n"
    f"{COMPARISON_PATH}"
)