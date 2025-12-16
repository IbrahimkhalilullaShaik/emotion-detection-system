import torch.nn as nn
from torchvision.models import resnet18

class ResNetEmotion(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.model = resnet18(weights="DEFAULT")
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)