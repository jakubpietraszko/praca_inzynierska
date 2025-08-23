import torch
import torch.nn as nn
import os
from torchvision.transforms import ToTensor, Compose, Resize
from torchvision import models

IDX_TO_LABEL = {
    0: "Cat",
    1: "Dog"
}

TRANSFORM = Compose([
    Resize((224, 224)),
    ToTensor()
])


class dogsVsCatsModel(nn.Module):
    def __init__(self, pretrained=True, freeze_features=True):
        super(dogsVsCatsModel, self).__init__()
        
        self.mobilenet = models.mobilenet_v2(pretrained=pretrained)
        
        if freeze_features:
            for param in self.mobilenet.features.parameters():
                param.requires_grad = False
        
        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 2)
        )
    
    def forward(self, x):
        return self.mobilenet(x)

def loadModel():
    model = dogsVsCatsModel()
    model_path = os.path.join(
        os.path.dirname(__file__), 'best_model.pth'
    )
    model.load_state_dict(torch.load(model_path))
    return model, model.mobilenet.features[2].conv[2], TRANSFORM, IDX_TO_LABEL, (3, 224, 224)