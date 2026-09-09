import torch
from torchvision.models import ( 
    resnet50,
    ResNet50_Weights,
    densenet121,
    DenseNet121_Weights,
    swin_t,
    Swin_T_Weights,
    vit_b_16,
    ViT_B_16_Weights,
)

from .data import TISSUE_CLASSES

def build_resnet50():
    weights = ResNet50_Weights.DEFAULT

    model = resnet50(weights=weights)

    model.fc = torch.nn.Linear(
        model.fc.in_features,
        len(TISSUE_CLASSES)
    )

    return model

def build_densenet121():
    weights = DenseNet121_Weights.DEFAULT

    model = densenet121(weights=weights)

    model.classifier = torch.nn.Linear(
        model.classifier.in_features,
        len(TISSUE_CLASSES),
    )

    return model

def build_swin_t():
    weights = Swin_T_Weights.DEFAULT
    model = swin_t(weights=weights)

    model.head = torch.nn.Linear(
        model.head.in_features,
        len(TISSUE_CLASSES),
    )

    return model

def build_vit_b_16():
    weights = ViT_B_16_Weights.DEFAULT
    model = vit_b_16(weights=weights)

    model.heads.head = torch.nn.Linear(
        model.heads.head.in_features,
        len(TISSUE_CLASSES),
    )

    return model