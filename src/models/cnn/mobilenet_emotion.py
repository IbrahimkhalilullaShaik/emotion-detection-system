import torch.nn as nn
from torchvision.models import mobilenet_v2

class MobileNetEmotion(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.model = mobilenet_v2(weights="DEFAULT")

        for param in self.model.features.parameters():
            param.requires_grad = False   # Freeze backbone

        self.model.classifier[1] = nn.Linear(1280, num_classes)

    def forward(self, x):
        return self.model(x)