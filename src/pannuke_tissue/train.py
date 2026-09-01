import torch
import csv
from datetime import datetime

from tqdm import tqdm

def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    epoch,
    batch_log_path=None,
    log_interval=10
):
    model.train()

    running_loss = 0.0

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch} - Training",
        unit="batch",
        position=1,
        leave=False,
    )

    for batch_idx, (images, labels) in enumerate(progress_bar):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        progress_bar.set_postfix(
            batch_loss=f"{loss.item():.4f}"
        )

        if (
            batch_log_path is not None
            and (batch_idx + 1) % log_interval == 0
        ):
            with open(batch_log_path, "a", newline="") as f:
                writer = csv.writer(f)

                writer.writerow([
                    datetime.now().isoformat(timespec="seconds"),
                    epoch,
                    batch_idx + 1,
                    loss.item()
                ])

    average_loss = running_loss / len(train_loader)

    return average_loss