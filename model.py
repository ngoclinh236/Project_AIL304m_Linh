import torch.nn as nn
from torchvision import models


class DogBreedAlexNet(nn.Module):
    def __init__(self, num_classes=120, pretrained=True):
        super().__init__()

        weights = models.AlexNet_Weights.DEFAULT if pretrained else None
        self.backbone = models.alexnet(weights=weights)

        
        for param in self.backbone.parameters():
            param.requires_grad = True

        
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9216, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)