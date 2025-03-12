import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))

from data.download_data import *
from data.make_data import *
from utils.functions import *
from models.model import *
import torch.nn as nn
import torch.optim as optim

epochs = 3
criterion = nn.CrossEntropyLoss()


# full-fine tuning ResNet (Gray_Scale)

dataset = make_gray_datas

train_loader, val_loader, test_loader = return_dataloader(dataset)
model = ResNet_full_gray().cuda()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


for i in range(epochs):
    train_model(model, train_loader, val_loader, criterion, optimizer)

visualize(model, test_loader, criterion, optimizer)


# # full-fine tuning DenseNet (Gray_Scale)
# train_loader, val_loader, test_loader = return_datalodader(make_gray_datas())
# model = DenseNet_full_gray.cuda()
# optimizer = optim.Adam(model.parameters(), lr=0.0001)

# for i in range(epochs):
#     train_model(model, train_loader, val_loader, criterion, optimizer)

# visualize(model, test_loader, criterion, optimizer)


# # partial-fine tuning ResNet (Gray_Scale)
# train_loader, val_loader, test_loader = return_datalodader(make_gray_datas())
# model = ResNet_partial_gray.cuda()
# optimizer = optim.Adam(model.parameters(), lr=0.0001)

# for i in range(epochs):
#     train_model(model, train_loader, val_loader, criterion, optimizer)

# visualize(model, test_loader, criterion, optimizer)


# # partial-fine tuning DenseNet (Gray_Scale)
# train_loader, val_loader, test_loader = return_datalodader(make_gray_datas())
# model = DenseNet_partial_gray.cuda()
# optimizer = optim.Adam(model.parameters(), lr=0.0001)

# for i in range(epochs):
#     train_model(model, train_loader, val_loader, criterion, optimizer)

# visualize(model, test_loader, criterion, optimizer)
