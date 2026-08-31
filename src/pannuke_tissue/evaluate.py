import torch

def evaluate(
    model, 
    val_loader,
    criterion,
    device,
) : 
    model.eval()

    running_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

        average_loss = running_loss / len(val_loader)

    return average_loss
