import torchvision.models as models
import torch.nn as nn

def get_efficientat(num_classes=2):
    model = models.mobilenet_v3_small(weights=None)
    model.features[0][0] = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    return model
