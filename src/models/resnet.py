import torchvision.models as models
import torch
import torch.nn as nn

def get_resnet18_pretrained(num_classes=2):
    #Gets pretrained weights
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    #Collapses input 3-channel layer to 1-channel layer
    original_weights = model.conv1.weight.data
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2, 2), padding=(3, 3), bias=False)
    model.conv1.weight.data = torch.mean(original_weights, dim=1, keepdim=True)

    #Modifies classification head to have classes = num_classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
    