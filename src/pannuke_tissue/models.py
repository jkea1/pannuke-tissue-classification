import torch
from torchvision.models import resnet50, ResNet50_Weights

from .data import TISSUE_CLASSES

def build_resnet50():
    weights = ResNet50_Weights.DEFAULT

    model = resnet50(weights=weights)

    model.fc = torch.nn.Linear(
        model.fc.in_features,
        len(TISSUE_CLASSES)
    )

    return model