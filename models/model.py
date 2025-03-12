from torchvision import models
import torch.nn as nn

# Function to create a ResNet34 model with grayscale input (full training)
def ResNet_full_gray():
    ResNet = models.resnet34(weights=None)  # Load ResNet34 without pre-trained weights
    ResNet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Modify first convolution layer for 1-channel (grayscale) input
    ResNet.fc = nn.Linear(in_features=512, out_features=2, bias=True)  # Modify the final fully connected layer to output 2 classes
    return ResNet

# Function to create a ResNet34 model with grayscale input (using pre-trained weights)
def ResNet_partial_gray():
    ResNet = models.resnet34(weights='ResNet34_Weights.DEFAULT')  # Load ResNet34 with pre-trained weights
    ResNet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Modify first convolution layer for 1-channel (grayscale) input
    ResNet.fc = nn.Linear(in_features=512, out_features=2, bias=True)  # Modify the final fully connected layer to output 2 classes
    return ResNet

# Function to create a DenseNet121 model with grayscale input (full training)
def DenseNet_full_gray():
    DenseNet = models.densenet121(weights=None)  # Load DenseNet121 without pre-trained weights
    DenseNet.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  # Modify first convolution layer for 1-channel (grayscale) input
    DenseNet.classifier = nn.Linear(in_features=1024, out_features=2, bias=True)  # Modify the final classifier layer to output 2 classes
    return DenseNet

# Function to create a DenseNet121 model with grayscale input (using pre-trained weights)
def DenseNet_partial_gray():
    DenseNet = models.densenet121(weights='DenseNet121_Weights.DEFAULT')  # Load DenseNet121 with pre-trained weights
    DenseNet.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  # Modify first convolution layer for 1-channel (grayscale) input
    DenseNet.classifier = nn.Linear(in_features=1024, out_features=2, bias=True)  # Modify the final classifier layer to output 2 classes
    return DenseNet

# Function to create a ResNet34 model with RGB input (full training)
def ResNet_full_RGB():
    ResNet = models.resnet34(weights=None)  # Load ResNet34 without pre-trained weights
    ResNet.fc = nn.Linear(in_features=512, out_features=2, bias=True)  # Modify the final fully connected layer to output 2 classes
    return ResNet

# Function to create a ResNet34 model with RGB input (using pre-trained weights)
def ResNet_partial_RGB():
    ResNet = models.resnet34(weights='ResNet34_Weights.DEFAULT')  # Load ResNet34 with pre-trained weights
    ResNet.fc = nn.Linear(in_features=512, out_features=2, bias=True)  # Modify the final fully connected layer to output 2 classes
    return ResNet

# Function to create a DenseNet121 model with RGB input (full training)
def DenseNet_full_RGB():
    DenseNet = models.densenet121(weights=None)  # Load DenseNet121 without pre-trained weights
    DenseNet.classifier = nn.Linear(in_features=1024, out_features=2, bias=True)  # Modify the final classifier layer to output 2 classes
    return DenseNet

# Function to create a DenseNet121 model with RGB input (using pre-trained weights)
def DenseNet_partial_RGB():
    DenseNet = models.densenet121(weights='DenseNet121_Weights.DEFAULT')  # Load DenseNet121 with pre-trained weights
    DenseNet.classifier = nn.Linear(in_features=1024, out_features=2, bias=True)  # Modify the final classifier layer to output 2 classes
    return DenseNet

# Placeholder functions for cropped versions, these return the corresponding full or partial models
def ResNet_full_crop():
    return ResNet_full_RGB  # Return ResNet model for RGB with full training

def ResNet_partial_crop():
    return ResNet_partial_RGB  # Return ResNet model for RGB with pre-trained weights

def DenseNet_full_crop():
    return DenseNet_full_RGB  # Return DenseNet model for RGB with full training

def DenseNet_partial_crop():
    return DenseNet_partial_RGB  # Return DenseNet model for RGB with pre-trained weights
