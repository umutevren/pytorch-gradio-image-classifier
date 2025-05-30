import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


class ExampleModel(nn.Module):
    def __init__(self, num_classes):
        super(ExampleModel, self).__init__()
        self.base_model = timm.create_model("efficientnet_b0", pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        network_out_size = 1280

        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(network_out_size, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output
