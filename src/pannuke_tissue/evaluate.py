import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support
)

from src.pannuke_tissue.data import TISSUE_CLASSES

def evaluate(
    model, 
    val_loader,
    criterion,
    device,
    return_predictions=False,
) : 
    model.eval()

    running_loss = 0.0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size

            predictions = outputs.argmax(dim=1) # the class ID with the highest logit

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.cpu().tolist()
            )

        average_loss = running_loss / len(val_loader.dataset)

        accuracy = accuracy_score(
            all_labels,
            all_predictions,
        )

        # Macro-F1
        # calculates F1 seperately for each of the 19 tissue classes, then gives every class equal importance when averaging
        macro_f1 = f1_score(
            all_labels,
            all_predictions,
            average="macro",
        )

        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels,
            all_predictions,
            labels=list(range(len(TISSUE_CLASSES))),
            zero_division=0,
        )

        per_class_metrics = {}

        for i, class_name in enumerate(TISSUE_CLASSES):
            per_class_metrics[class_name] = {
                "precision": precision[i],
                "recall": recall[i],
                "f1": f1[i],
                "support": support[i],
            }

    if return_predictions:
        return average_loss, accuracy, macro_f1, per_class_metrics, all_labels, all_predictions

    return average_loss, accuracy, macro_f1, per_class_metrics