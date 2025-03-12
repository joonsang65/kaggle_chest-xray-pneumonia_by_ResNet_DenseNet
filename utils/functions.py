import torch
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def test_model(model, dataloader):   # confusion matrix 계산을 위해 함수 분리
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
            hit += torch.sum((predicted == labels) & (labels == 1)).item()  # 적중
            miss += torch.sum((predicted != labels) & (labels == 1)).item()  # 탈루
            FA += torch.sum((predicted != labels) & (labels == 0)).item()  # 오경보
            CR += torch.sum((predicted == labels) & (labels == 0)).item()  # 정기각
    print(f"test Accuracy: {100 * correct / total:.2f}%")
    return hit, FA, miss, CR  # 계산한 값 return


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

    # 현재 스크립트가 있는 디렉토리 기준으로 상대 경로 설정 / 필요 시 수정
    BASE_DIR =os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    BASE_DIR = os.path.join(BASE_DIR, "chest_xray")
    
    DATA_DIR = os.path.join(BASE_DIR, "chest_xray")

    # 데이터셋 경로 설정 (상대 경로) / 필요 시 수정
    train_path = os.path.join(DATA_DIR, "train")
    val_path = os.path.join(DATA_DIR, "val")
    test_path = os.path.join(DATA_DIR, "test")

    # 데이터셋 생성
    train_dataset = datas_class(train_path)
    val_dataset = datas_class(val_path)
    test_dataset = datas_class(test_path)

    # DataLoader 설정
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