import os
import random
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import torch

class make_gray_datas(Dataset):  # Define class for batch-wise preprocessing to avoid high memory usage
    def __init__(self, path):
        N_path = os.path.join(path, 'NORMAL')
        P_path = os.path.join(path, 'PNEUMONIA')
        # Data is divided into NORMAL and PNEUMONIA folders, classify accordingly
        self.images = []
        self.labels = []

        for p in os.listdir(P_path):
            self.images.append(os.path.join(P_path, p))
            self.labels.append(1)  # Label 1 for pneumonia

        for n in os.listdir(N_path):
            self.images.append(os.path.join(N_path, n))
            self.labels.append(0)  # Label 0 for normal

        # Shuffle the dataset (NORMAL and PNEUMONIA images)
        data = list(zip(self.images, self.labels))
        random.shuffle(data)
        self.images, self.labels = zip(*data)

        self.transform = transforms.Compose([
                        transforms.Resize((224, 224)),  # Resize image
                        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
                        transforms.ToTensor(),  # Normalize
                    ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path)

        label = self.labels[idx]

        image = self.transform(image)
        image = image.float()  # Ensure correct data format

        return image, torch.tensor(label)


class make_RGB_datas(Dataset):  # Convert grayscale images to RGB format
    def __init__(self, path):
        N_path = os.path.join(path, 'NORMAL')
        P_path = os.path.join(path, 'PNEUMONIA')
        self.images = []
        self.labels = []

        for p in os.listdir(P_path):
            self.images.append(os.path.join(P_path, p))
            self.labels.append(1)

        for n in os.listdir(N_path):
            self.images.append(os.path.join(N_path, n))
            self.labels.append(0)

        self.transform = transforms.Compose([
                        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
                        transforms.Resize((224, 224)),
                        transforms.ToTensor()
                    ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path)

        label = self.labels[idx]

        image = self.transform(image)
        image = image.float()

        return image, torch.tensor(label)


class make_crop_datas(Dataset):  # Apply cropping on images
    def __init__(self, path):
        N_path = os.path.join(path, 'NORMAL')
        P_path = os.path.join(path, 'PNEUMONIA')

        self.images = []
        self.labels = []

        for p in os.listdir(P_path):
            self.images.append(os.path.join(P_path, p))
            self.labels.append(1)

        for n in os.listdir(N_path):
            self.images.append(os.path.join(N_path, n))
            self.labels.append(0)

        # Shuffle images (not required, but useful for CAM visualization)
        data = list(zip(self.images, self.labels))
        random.shuffle(data)
        self.images, self.labels = zip(*data)

        self.transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.CenterCrop((180, 180)),  # Crop to 180x180 pixels
                        transforms.Grayscale(num_output_channels=1),
                        transforms.ToTensor(),
                    ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path)

        label = self.labels[idx]

        image = self.transform(image)
        image = image.float()  # Ensure correct data format

        return image, torch.tensor(label)
