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
    def __init__(self, weights='DEFAULT', freeze_features=True):
        super(dogsVsCatsModel, self).__init__()
        
        self.mobilenet = models.mobilenet_v3_small(weights=weights)
        
        if freeze_features:
            for param in self.mobilenet.features.parameters():
                param.requires_grad = False
        
        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(576, 2)
        )
    
    def forward(self, x):
        return self.mobilenet(x)

def loadModel():
    model = dogsVsCatsModel()
    model_path = os.path.join(
        os.path.dirname(__file__), 'best_model.pth'
    )
    model.load_state_dict(torch.load(model_path))
    return model, model.mobilenet.features[12][0], TRANSFORM, LABEL_TO_IDX, (3, 224, 224)