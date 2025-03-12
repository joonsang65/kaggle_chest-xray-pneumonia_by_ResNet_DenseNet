import torch
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data.make_data import *
from models.model import *

def test_model(model, dataloader):   # Separate function for confusion matrix calculation
    model.eval()
    correct = 0
    total = 0
    hit, FA, miss, CR = 0, 0, 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            hit += torch.sum((predicted == labels) & (labels == 1)).item()  # Hit
            miss += torch.sum((predicted != labels) & (labels == 1)).item()  # Miss
            FA += torch.sum((predicted != labels) & (labels == 0)).item()  # False Alarm
            CR += torch.sum((predicted == labels) & (labels == 0)).item()  # Correct Rejection
    print(f"test Accuracy: {100 * correct / total:.2f}%")
    return hit, FA, miss, CR  # Return calculated values


def validate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def train_model(model, train_loader, val_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    with tqdm(train_loader, unit="batch") as tepoch:
        for inputs, labels in tepoch:
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            tepoch.set_postfix(loss=running_loss/len(train_loader), accuracy=100 * correct/total)

    print(f"Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct/total:.2f}%")

    val_accuracy = validate_model(model, val_loader)
    print(f"Validation Accuracy: {val_accuracy:.2f}%")


def return_dataloader(datas_class):

    # Set relative path based on the directory of the current script / Modify if necessary
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    BASE_DIR = os.path.join(BASE_DIR, "chest_xray")

    DATA_DIR = os.path.join(BASE_DIR, "chest_xray")

    # Set dataset paths (relative paths) / Modify if necessary
    train_path = os.path.join(DATA_DIR, "train")
    val_path = os.path.join(DATA_DIR, "val")
    test_path = os.path.join(DATA_DIR, "test")

    # Create datasets
    train_dataset = datas_class(train_path)
    val_dataset = datas_class(val_path)
    test_dataset = datas_class(test_path)

    # Configure DataLoader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    print(f"Train Path: {train_path}")
    print(f"Validation Path: {val_path}")
    print(f"Test Path: {test_path}")

    return train_loader, val_loader, test_loader


def visualize(model, loader, criterion, optimizer):
    hit, FA, miss, CR = test_model(model, loader, criterion, optimizer)

    precision = hit/(hit+FA)
    recall = hit/(hit+miss)
    f1_score = 2 * (precision * recall) / (precision + recall)

    print(f'precision : {precision:-2f}')
    print(f'recall : {recall:-2f}')
    print(f'f1_score : {f1_score:-2f}\n')

    conf_matrix = np.array([[hit, FA],
                            [miss, CR]])

    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d',
                xticklabels=['Actual Pneumonia', 'Actual Normal'],
                yticklabels=['Predicted Pneumonia', 'Predicted Normal'])

    plt.title('Confusion Matrix')
    plt.show()


# Function to minimize repetitive input
def get_input(prompt, options):
    while True:
        value = input(prompt)
        if value in options:
            return value
        print("===================================================")
        print(f"Please enter one of {options}")


# Function to define model and dataset for main.py
def define_model():
    ans = 'n'
    while ans != 'y':
        crop = get_input("Crop model has only RGB images.\nDo you want to crop the images? (y/n): ", ['y', 'n'])
        tuning = get_input("Do you want to fine-tune the model? (Full/partial): ", ['Full', 'partial'])
        model_type = get_input("Which model do you want to use? (ResNet/DenseNet): ", ['ResNet', 'DenseNet'])

        # If crop='y', do not use Grayscale
        Grayscale = None if crop == 'y' else get_input("Do you want to convert the images to grayscale? (y/n): ", ['y', 'n'])

        # Data function mapping
        data_funcs = {
            'y': make_crop_datas,
            'n': {'y': make_gray_datas, 'n': make_RGB_datas}
        }

        # Model function mapping
        model_funcs = {
            'y': {
                'Full': {'ResNet': ResNet_full_crop, 'DenseNet': DenseNet_full_crop},
                'partial': {'ResNet': ResNet_partial_crop, 'DenseNet': DenseNet_partial_crop}
            },
            'n': {
                'y': {
                    'Full': {'ResNet': ResNet_full_gray, 'DenseNet': DenseNet_full_gray},
                    'partial': {'ResNet': ResNet_partial_gray, 'DenseNet': DenseNet_partial_gray}
                },
                'n': {
                    'Full': {'ResNet': ResNet_full_RGB, 'DenseNet': DenseNet_full_RGB},
                    'partial': {'ResNet': ResNet_partial_RGB, 'DenseNet': DenseNet_partial_RGB}
                }
            }
        }

        # Select data processing function (access differently based on crop choice)
        datas = data_funcs[crop] if crop == 'y' else data_funcs['n'][Grayscale]

        # Select model (if crop='y', Grayscale is not applicable)
        if crop == 'y':
            model = model_funcs['y'][tuning][model_type]()
        else:
            model = model_funcs['n'][Grayscale][tuning][model_type]()

        print(f"You selected\nModel: {model_type}\nTuning: {tuning}\nCrop: {crop}\nGrayscale: {Grayscale}")
        while True:
            ans = input("\n Are these choices correct? (y/n): ")
            if ans not in ['y', 'n']:
                print("Please enter 'y' or 'n'")
            elif ans == 'n':
                print("===================================================")
                print("Please re-enter the information")
                continue
            else:
                break

    return datas, model
