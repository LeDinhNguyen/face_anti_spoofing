import torch
from torch import nn
from torchvision import models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MyMobileNetV2(nn.Module):
    def __init__(self, num_classes_fc=128, num_classes_output=2):
        super(MyMobileNetV2, self).__init__()

        # Load the pre-trained MobileNetV2 model
        mobilenet_v2 = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        # Modify the classifier for your specific task
        num_ftrs = mobilenet_v2.classifier[1].in_features
        mobilenet_v2.classifier[1] = nn.Linear(num_ftrs, num_classes_fc)
        # print(mobilenet_v2.classifier[1])

        # Fully connected network
        fully_connected_network = nn.Sequential(
            nn.Linear(num_classes_fc, num_classes_fc),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(num_classes_fc, num_classes_output)
        )

        # Combine MobileNetV2 and the fully connected network
        self.model = nn.Sequential(
            mobilenet_v2,
            # nn.AdaptiveAvgPool2d(1),
            # nn.Flatten(),
            fully_connected_network
        )

    def forward(self, x):
        return self.model(x)