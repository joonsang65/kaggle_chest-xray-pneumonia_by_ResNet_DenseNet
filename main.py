# import sys, os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))

# # from data.download_data import *
from data.make_data import *
from utils.functions import *
from models.model import *
import torch.nn as nn
import torch.optim as optim

data, model = define_model()

print(data)

print(model)
train_loader, val_loader, test_loader = return_dataloader(data)

print(len(train_loader))


# model = model().cuda()
# epochs = 3
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0001)

# for i in range(epochs):
#     train_model(model, train_loader, val_loader, criterion, optimizer)

# visualize(model, test_loader, criterion, optimizer)