import torch.nn as nn
from torchvision import models

class DefectCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(weights="IMAGENET1K_V1")
        self.model.fc = nn.Linear(512, 1)

    def forward(self, x):
        return self.model(x)
