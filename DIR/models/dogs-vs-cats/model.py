import torch
import torch.nn as nn
import os
from torchvision.transforms import ToTensor, Compose, Resize

IDX_TO_LABEL = {
    0: "Cat",
    1: "Dog"
}

TRANSFORM = Compose([
    Resize((224, 224)),
    ToTensor()
])

class dogsVsCatsModel(nn.Module):
    def __init__(self):
        super(dogsVsCatsModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.reshape(-1, 64 * 28 * 28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
def loadModel():
    model = dogsVsCatsModel()
    model_path = os.path.join(
        os.path.dirname(__file__), 'best_model.pth'
    )
    model.load_state_dict(torch.load(model_path))
    return model, list(model.modules())[0].conv3, TRANSFORM, IDX_TO_LABEL, (3, 224, 224)