import torch
import torchvision.models as models
from torch import nn


class Classifier(nn.Module):
    def __init__(
        self, num_classes: int = 2, backbone: str = "mobilenet_v3_large", pretrained: bool = True
    ):
        super().__init__()
        self.model = models.__dict__[backbone](pretrained=pretrained)
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

        # freeze all layers except the last one
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)
