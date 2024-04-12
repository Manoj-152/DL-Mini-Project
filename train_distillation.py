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
import random

torch.manual_seed(42)
random.seed(10)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# device = torch.device("cuda:0")
num_epochs = 300
batch_size = 512
learning_rate = 0.01

# train_dataset = Cifar10Dataset(split='train')
# test_dataset = Cifar10Dataset(split='test')
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

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
plt.imshow(pic[5].permute(1,2,0))
plt.savefig('scrap.png')
# exit()

teacher = Resnet(BasicBlock, [3,4,6,3], 10)
teacher = teacher.to(device)
teacher_dict = torch.load('teacher_checkpoint.pth')
print(f"Teacher Test Accuracy: {teacher_dict['accuracy']:.2f}%")
teacher.load_state_dict(teacher_dict['model'])
student = Resnet(BasicBlock, [2,1,1,1], 10)
student = student.to(device)

num_params_teacher = np.sum([p.nelement() for p in teacher.parameters()]) # Printing the number of parameters
num_params_student = np.sum([p.nelement() for p in student.parameters()])
print('Teacher:', num_params_teacher, ' parameters')
print('Student:', num_params_student, ' parameters')

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(student.parameters(), lr=learning_rate)
# optimizer = optim.SGD(student.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

def train_distillation(student, teacher, criterion, optimizer, num_epochs, T=2, soft_target_loss_weight=0.25, ce_loss_weight=0.75):
    best_acc = 0
    teacher.eval()
    for epoch in range(num_epochs):
        student.train()
        running_loss = 0.0
        running_ce_loss = 0.0
        running_distil_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm(train_loader):
            # images, labels = images.to(device), labels.to(device)
            images, labels = images.to(device), labels.to(device)
            # if labels.dim() > 1:
            #     labels = torch.max(labels, 1)[1] 

            # Forward pass
            with torch.no_grad():
                teacher_logits = teacher(images)
            student_logits = student(images)

            soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
            log_soft_targets = nn.functional.log_softmax(teacher_logits / T, dim=-1)
            soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)
            # print(torch.sum(soft_targets.log() - soft_prob))
            # print(torch.sum(soft_targets * (log_soft_targets - soft_prob)))
            soft_targets_loss = torch.sum(soft_targets * (log_soft_targets - soft_prob)) / soft_prob.size()[0] * (T**2)
            
            ce_loss = criterion(student_logits, labels)
            loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * ce_loss

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_ce_loss += ce_loss.item()
            running_distil_loss += soft_targets_loss.item()

            _, predicted = torch.max(student_logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # break
        
        print(scheduler.get_lr())
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Total Loss: {epoch_loss:.4f}, Train Accuracy: {(100 * correct / total):.2f}%')
        print(f'CE Loss: {(running_ce_loss / len(train_loader)):.4f}, Distillation Loss: {(running_distil_loss / len(train_loader)):.4f}')

        # Validation accuracy
        student.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(test_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = student(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                # break

        acc = 100 * correct / total
        print(f'Accuracy of the model on the test images: {(100 * correct / total):.2f}%')
        if acc > best_acc:
            print("Saving best checkpoint")
            save_dict = {'model': student.state_dict(), 'best_epoch': epoch, 'accuracy': acc}
            torch.save(save_dict, 'best_ckpt_student_distil_new.pth')
            best_acc = acc

        scheduler.step()


train_distillation(student, teacher, criterion, optimizer, num_epochs)
# torch.save(model.state_dict(), 'mini_project_model.pth')