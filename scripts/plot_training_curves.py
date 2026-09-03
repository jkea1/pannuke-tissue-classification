from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

EXPERIMENT_NAME = "03_resnet50_augmentation_30ep_lr1e-4_bs32"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EPOCH_LOG_PATH = (
    PROJECT_ROOT / "outputs" / "metrics" / f"{EXPERIMENT_NAME}_epoch_metrics.csv"
)

FIGURE_DIR = (
    PROJECT_ROOT / "outputs" / "figures"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

df = pd.read_csv(EPOCH_LOG_PATH)

# training / val loss curves 
plt.figure()

plt.plot(
    df["epoch"],
    df["train_loss"],
    marker="o",
    label="Train Loss",
)

plt.plot(
    df["epoch"],
    df["val_loss"],
    marker="o",
    label="Validation Loss",
)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")

plt.legend()
plt.grid()

plt.savefig(
    FIGURE_DIR / f"{EXPERIMENT_NAME}_loss_curve.png",
    bbox_inches="tight",
)

# val macro-F1 curve
plt.figure()

plt.plot(
    df["epoch"],
    df["val_macro_f1"],
    marker="o",
    label="Validation Macro-F1",
)

plt.xlabel("Epoch")
plt.ylabel("Macro-F1")
plt.title("Validation Macro-F1")

plt.legend()
plt.grid()

plt.savefig(
    FIGURE_DIR / f"{EXPERIMENT_NAME}_macro_f1_curve.png",
    bbox_inches="tight",
)

plt.show()