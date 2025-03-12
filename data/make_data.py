import os
import random
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import torch



class make_gray_datas(Dataset):  # 전체 데이터를 한 번에 정제하면 메모리 요구량이 매우 커지기 때문에 batch 단위 정제를 위한 class 정의
    def __init__(self, path):
        N_path = os.path.join(path, 'NORMAL')
        P_path = os.path.join(path, 'PNEUMONIA')
        # 위의 경로 안에 정상 폴더와 폐렴 폴더로 나눠져 있어서 이를 기준으로 데이터 분리
        self.images = []
        self.labels = []

        for p in os.listdir(P_path):
            self.images.append(os.path.join(P_path, p))
            self.labels.append(1)  # 폐렴이면 1

        for n in os.listdir(N_path):
            self.images.append(os.path.join(N_path, n))
            self.labels.append(0)  # 정상이면 0

            # 구분한 label 기준으로 학습 시킬 예정

        # 정상과 폐렴 이미지 한 번 섞기
        data = list(zip(self.images, self.labels))
        random.shuffle(data)
        self.images, self.labels = zip(*data)

        self.transform = transforms.Compose([
                        transforms.Resize((224, 224)),  # 이미지 크기 조정
                        transforms.Grayscale(num_output_channels=1),  # 기본적으로 흑백 이미지라 흑백 처리
                        transforms.ToTensor(),  # 정규화
                    ])

        # 어차피 정제 과정은 똑같기 때문에 class 안에서 transform 정의

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path)

        label = self.labels[idx]

        image = self.transform(image)
        image = image.float()  # 데이터 형태 맞춰주기

        return image, torch.tensor(label)



class make_RGB_datas(Dataset):
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
                        transforms.Grayscale(num_output_channels=3),  # Grayscale to RGB
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



class make_crop_datas(Dataset):
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


        # 정상과 폐렴 이미지 한 번 섞기
        ## 굳이 할 필요 없는 과정이지만, CAM 결과 볼 때 0, 1 둘 다 확인하기 위함
        data = list(zip(self.images, self.labels))
        random.shuffle(data)
        self.images, self.labels = zip(*data)

        self.transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.CenterCrop((180,180)),  # 180 * 180 사이즈로 Crop
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
        image = image.float()  # 데이터 형태 맞춰주기

        return image, torch.tensor(label)