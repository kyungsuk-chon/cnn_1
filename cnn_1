import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

transform = transforms.Compose(
    [transforms.Resize((128,128)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.ImageFolder(root='./data/hybrid/train', transform=transform)
testset = torchvision.datasets.ImageFolder(root='./data/hybrid/test', transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=0)

classes = ('low', 'mid', 'top')

INPUT_DIM = 128
NUM_FILTERS = 32
KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1
NUM_CLASSES = 3
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10

import torch.nn as nn
import torch.nn.functional as F


class net(nn.Module):

    def __init__(self, input_dim, num_filters, kernel_size, stride, padding, num_classes):
        super(net, self).__init__()
        self.input_dim = input_dim
        conv_output_size = int((input_dim - kernel_size + 2 * padding) / stride) + 1  # conv layer output size
        pool_output_size = int((conv_output_size - kernel_size) / stride) + 1  # pooling layer output size

        self.conv = nn.Conv2d(3, num_filters, kernel_size=kernel_size, stride=stride, padding=padding)
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        self.relu = nn.ReLU()
        self.dense = nn.Linear(pool_output_size * pool_output_size * num_filters, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # resize to fit into final dense layer
        x = self.dense(x)
        return x

model = net(INPUT_DIM, NUM_FILTERS, KERNEL_SIZE, STRIDE, PADDING, NUM_CLASSES)
criterion = nn.CrossEntropyLoss()   # do not need softmax layer when using CEloss criterion
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

for i in range(15):
    print("start {}th epoch".format(i))
    temp_loss = []
    for (x, y) in trainloader:
        x, y = x.float(), y
        outputs = model(x)
        loss = criterion(outputs, y)
        temp_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Loss at {}th epoch: {}".format(i, np.mean(temp_loss)))

y_pred, y_true = [], []
with torch.no_grad():
  for x, y in testloader:
    x, y = x.float(), y
    outputs = model(x)
    _, predicted = torch.max(outputs.data,1)
    # outputs = F.softmax(model(x)).max(1)[-1]       # predicted label
    y_true += list(y.cpu().numpy())                # true label
    # y_pred += list(outputs.cpu().numpy())
    y_pred += list(predicted.cpu().numpy())

print(y_true)
print(y_pred)
# evaluation result
from sklearn.metrics import accuracy_score
print("accuracy score is : ")
print(accuracy_score(y_true, y_pred))
