from torchvision import models
import torch.nn as nn


def ResNet_full_gray():
    ResNet = models.resnet34(weights=None)
    ResNet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    ResNet.fc = nn.Linear(in_features=512, out_features=2, bias=True)
    return ResNet


def ResNet_partial_gray():
    ResNet = models.resnet34(weights= 'ResNet34_Weights.DEFAULT')
    ResNet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    ResNet.fc = nn.Linear(in_features=512, out_features=2, bias=True)
    return ResNet


def DenseNet_full_gray():
    DenseNet = models.densenet121(weights=None)
    DenseNet.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    DenseNet.classifier = nn.Linear(in_features=1024, out_features=2, bias=True)
    return DenseNet


def DenseNet_partial_gray():
    DenseNet = models.densenet121(weights = 'DenseNet121_Weights.DEFAULT')
    DenseNet.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    DenseNet.classifier = nn.Linear(in_features=1024, out_features=2, bias=True)
    return DenseNet


def ResNet_full_RGB():
    ResNet = models.resnet34(weights=None)
    ResNet.fc = nn.Linear(in_features=512, out_features=2, bias=True)
    return ResNet


def ResNet_partial_RGB():
    ResNet = models.resnet34(weights= 'ResNet34_Weights.DEFAULT')
    ResNet.fc = nn.Linear(in_features=512, out_features=2, bias=True)
    return ResNet


def DenseNet_full_RGB():
    DenseNet = models.densenet121(weights=None)
    DenseNet.classifier = nn.Linear(in_features=1024, out_features=2, bias=True)
    return DenseNet


def DenseNet_partial_RGB():
    DenseNet = models.densenet121(weights = 'DenseNet121_Weights.DEFAULT')
    DenseNet.classifier = nn.Linear(in_features=1024, out_features=2, bias=True)
    return DenseNet


def ResNet_full_crop():
    return ResNet_full_RGB

def ResNet_partial_crop():
    return ResNet_partial_RGB

def DenseNet_full_crop():
    return DenseNet_full_RGB

def DenseNet_partial_crop():
    return DenseNet_partial_RGB
