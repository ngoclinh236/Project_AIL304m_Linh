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

        # sửa classifier
        in_features = self.backbone.classifier[6].in_features
        self.backbone.classifier[6] = nn.Linear(in_features, num_classes)

        # freeze feature extractor
        for param in self.backbone.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.backbone(x)