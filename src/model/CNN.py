import torch
import torch.nn as nn

# Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1)
        )
        self.fc1 = nn.Linear(2803712, 64)
        self.norm1 = nn.LayerNorm(64)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(64, 4)
        self.norm2 = nn.LayerNorm(4)
        self.softmax = nn.Softmax(dim=1)
    # __init__
    def forward(self, x):
        out = self.layer1(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.norm1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.norm2(out)
        out = self.dropout(out)
        out = self.softmax(out)
        return out
    # forward
# NueralNet
