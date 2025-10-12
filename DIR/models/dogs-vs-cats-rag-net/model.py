import torch
import torch.nn as nn
import os
from torchvision.transforms import ToTensor, Compose, Resize
from torchvision import models

LABEL_TO_IDX = {
    'cat': 0,
    'dog': 1
}

TRANSFORM = Compose([
    Resize((224, 224)),
    ToTensor()
])

class dogsVsCatsModel(nn.Module):
    def __init__(self, pretrained=True, freeze_features=True):
        super(dogsVsCatsModel, self).__init__()

        self.regnet = models.regnet_y_400mf(pretrained=pretrained)
        if freeze_features:
            for param in self.regnet.parameters():
                param.requires_grad = False

        self.regnet.fc = nn.Linear(440, 2)

    def forward(self, x):
        x = self.regnet(x)
        return x


def loadModel():
    model = dogsVsCatsModel()
    model_path = os.path.join(
        os.path.dirname(__file__), 'best_model.pth'
    )
    model.load_state_dict(torch.load(model_path))
    return model, model.regnet.trunk_output.block4[-1].f.c[0], TRANSFORM, LABEL_TO_IDX, (3, 224, 224)