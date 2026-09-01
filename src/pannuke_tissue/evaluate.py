import torch
from sklearn.metrics import accuracy_score, f1_score

def evaluate(
    model, 
    val_loader,
    criterion,
    device,
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

            running_loss += loss.item()

            predictions = outputs.argmax(dim=1) # the class ID with the highest logit

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.cpu().tolist()
            )

        average_loss = running_loss / len(val_loader)

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

    return average_loss, accuracy, macro_f1
