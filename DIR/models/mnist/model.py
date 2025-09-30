import torch
import torch.nn as nn
import os
from torchvision.transforms import ToTensor, Compose, Resize, Grayscale


LABEL_TO_IDX = {
    'zero': 0,
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'seven': 7,
    'eight': 8,
    'nine': 9
}

TRANSFORM = Compose([
    Resize((28, 28)),
    Grayscale(num_output_channels=1),
    ToTensor()
])

class mnistModel(nn.Module):
    def __init__(self):
        super(mnistModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
    
def loadModel():
    model = mnistModel()
    model_path = os.path.join(os.path.dirname(__file__), 'best_model.pth')
    model.load_state_dict(torch.load(model_path))
    return model, list(model.modules())[0].conv2, TRANSFORM, LABEL_TO_IDX, (1, 28, 28)
