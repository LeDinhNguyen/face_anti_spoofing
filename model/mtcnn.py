import torch
from torch import nn
from facenet_pytorch import MTCNN, InceptionResnetV1

class FaceAntiSpoofingModel(nn.Module):
    def __init__(self):
        super(FaceAntiSpoofingModel, self).__init__()

        # Fully connected layers for classification
        self.fc1 = nn.Linear(512, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)  # Output has 2 classes (live or spoof)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x