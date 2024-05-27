import torch
import torchvision
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from model.net import LeNet5

train_dataset = datasets.MNIST(root='../data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='../data', train=False, transform=transforms.ToTensor())

# 定义数据加载器
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)

model = LeNet5()
loss_fun = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

train_loss = []
test_acc = []

model.cuda()
loss_fun.cuda()

# 训练模型
model.train()
for epoch in tqdm(range(10)):
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        loss = loss_fun(outputs, labels)
        train_loss.append(loss.data.cpu().numpy())
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, 10, i + 1, len(train_loader),
                                                                     loss.item()))

    # 每个epoch进行一次eval
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            # torch.max(input, dim, max=None, max_indices=None) -> (Tensor, LongTensor)
            _, predicted = torch.max(outputs.data, 1)  # 返回行最大值，和对应的行标
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_acc.append(100 * correct / total)
        print('Epoch [{}/{}], Test Accuracy:{:.2f}%'.format(epoch + 1, 10, 100 * correct / total))
