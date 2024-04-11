import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dataloader import Cifar10Dataset
import resnet
from resnet import Resnet, BasicBlock
import numpy as np
from tqdm import tqdm
import torchvision
import matplotlib.pyplot as plt

torch.manual_seed(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# device = torch.device("cuda:0")
num_epochs = 100
batch_size = 512
learning_rate = 0.01

# train_dataset = Cifar10Dataset(split='train')
# test_dataset = Cifar10Dataset(split='test')
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

pic, label = next(iter(train_loader))
plt.imshow(pic[2].permute(1,2,0))
plt.savefig('scrap.png')
exit()

model = Resnet(BasicBlock, [2,1,1,1], 10)
# model.to(device)
model = model.cuda()
num_params = np.sum([p.nelement() for p in model.parameters()]) # Printing the number of parameters
print(num_params, ' parameters')

criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

def train(model, criterion, optimizer, num_epochs):
    best_acc = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm(train_loader):
            # images, labels = images.to(device), labels.to(device)
            images, labels = images.cuda(), labels.cuda()
            # if labels.dim() > 1:
            #     labels = torch.max(labels, 1)[1] 

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
            # break
        
        print(scheduler.get_lr())
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Train Accuracy: {(100 * correct / total):.2f}%')

        # Validation accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(test_loader):
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                # break

        acc = 100 * correct / total
        print(f'Accuracy of the model on the test images: {(100 * correct / total):.2f}%')
        if acc > best_acc:
            print("Saving best checkpoint")
            save_dict = {'model': model.state_dict(), 'best_epoch': epoch, 'accuracy': acc}
            torch.save(save_dict, 'best_ckpt.pth')
            best_acc = acc

        scheduler.step()


train(model, criterion, optimizer, num_epochs)
# torch.save(model.state_dict(), 'mini_project_model.pth')