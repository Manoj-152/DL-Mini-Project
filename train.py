import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import resnet
from resnet import Resnet, BasicBlock
import numpy as np
from tqdm import tqdm
import torchvision
import matplotlib.pyplot as plt
import random
import os

# Seeding for consistency
torch.manual_seed(42)
random.seed(10)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Current Device:",device)

num_epochs = 200
batch_size = 512
learning_rate = 0.01

# Setting up the data augmentation transforms
# --> Random Cropping
# --> Random Brightness Changer
# --> Random Grayscale (with 0.2 probability)
# --> Random Horizontal flip (with 0.5 probability)
# --> Random occlusion of images (done using random erasing)

crop = transforms.RandomCrop(32, padding=4)
rand_crop = transforms.Lambda(lambda x: crop(x) if random.random() < 0.75 else x)
color_changer = transforms.ColorJitter(brightness=0.25)
rand_color = transforms.Lambda(lambda x: color_changer(x) if random.random() < 0.5 else x)
grayscale = transforms.Grayscale(num_output_channels=3)
rand_grayscale = transforms.Lambda(lambda x: grayscale(x) if random.random() < 0.2 else x)

transform_train = transforms.Compose([
    rand_color,
    rand_crop,
    transforms.RandomHorizontalFlip(0.5),
    rand_grayscale,
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.RandomErasing(p=0.75, scale=(0.02, 0.33)),
])

# No augmentation on validation dataset except normalization
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Loading the dataset and creating the dataloader
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

# Debugging purposes
# pic, label = next(iter(train_loader))
# plt.imshow(pic[5].permute(1,2,0))
# plt.savefig('scrap.png')
# exit()

# Loading the Resnet-9 model
model = Resnet(BasicBlock, [2,1,1,1], 10)
model = model.to(device)
num_params = np.sum([p.nelement() for p in model.parameters()]) # Printing the number of parameters
print(num_params, ' parameters')

# Setting up the loss function, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

os.makedirs("checkpoints", exist_ok = True)

def train(model, criterion, optimizer, num_epochs):
    best_acc = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        print("Current LR: ", scheduler.get_lr())
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Train Accuracy: {(100 * correct / total):.2f}%')

        # Validation accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(test_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f'Accuracy of the model on the test images: {(100 * correct / total):.2f}%')

        # Saving the best checkpoint
        if acc > best_acc:
            print("Saving best checkpoint")
            save_dict = {'model': model.state_dict(), 'best_epoch': epoch, 'accuracy': acc}
            torch.save(save_dict, 'checkpoints/best_ckpt_augmentation.pth')
            best_acc = acc

        # Stepping the scheduler after every epoch completion
        scheduler.step()


train(model, criterion, optimizer, num_epochs)