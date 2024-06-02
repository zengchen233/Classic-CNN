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
from torchvision.transforms.functional import InterpolationMode
import matplotlib.pyplot as plt

from model.net import AlexNet

transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    transforms.Resize((227, 227), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.2435, 0.2616]),
])

train_dataset = datasets.CIFAR10(root='../data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='../data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=256, shuffle=False)

model = AlexNet(num_classes=10).cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)

train_loss, train_acc, test_loss, test_acc = [], [], [], []
best_val_loss = float('inf')
save_path = './AlexNet.pth'

num_epochs = 20
for epoch in tqdm(range(num_epochs)):
    model.train()
    temp_loss, temp_correct = 0, 0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        temp_correct += (predicted == labels).sum().item()
        temp_loss += loss
    print(
        'Epoch {}|{}, Train Loss:{:.4f}, Train Acc:{:.2f}%'.format(epoch + 1, num_epochs, temp_loss / len(train_loader),
                                                                   temp_correct / len(train_dataset) * 100))
    train_loss.append(temp_loss / len(train_loader))
    train_acc.append(temp_correct / len(train_dataset))

    temp_loss, temp_correct = 0, 0
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            temp_correct += (predicted == labels).sum().item()
            temp_loss += loss
    print('Epoch {}|{}, Test Loss:{:.4f}, Test Acc:{:.2f}%'.format(epoch + 1, num_epochs, temp_loss / len(test_loader),
                                                                   temp_correct / len(test_dataset) * 100))
    test_loss.append(temp_loss / len(test_loader))
    test_acc.append(temp_correct / len(test_dataset))

    if temp_loss < best_val_loss:
        best_val_loss = temp_loss
        # torch.save(model.state_dict(), save_path)
