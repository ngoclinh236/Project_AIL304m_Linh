import torch.nn as nn
from torchvision import models


class DogBreedAlexNet(nn.Module):
    def __init__(self, num_classes=120, pretrained=True):
        super().__init__()

        if pretrained:
            weights = models.AlexNet_Weights.DEFAULT
        else:
            weights = None

        self.backbone = models.alexnet(weights=weights)

        # ===== Freeze phần đầu =====
        for param in self.backbone.features[:6].parameters():
            param.requires_grad = False

        # ===== Unfreeze phần sau =====
        for param in self.backbone.features[6:].parameters():
            param.requires_grad = True

        # =====classifier =====
        in_features = self.backbone.classifier[6].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9216, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)