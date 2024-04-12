import pickle5 as pickle
import argparse
import numpy as np
from resnet import Resnet, BasicBlock
import torch
from torchvision import transforms
import pandas as pd

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Implement argparse here

test_dict = unpickle('data/cifar_test_nolabels.pkl')
test_data, test_ids = test_dict[b'data'], test_dict[b'ids']

img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

model = Resnet(BasicBlock, [2,1,1,1], 10)
# model.to(device)
model = model.cuda()
num_params = np.sum([p.nelement() for p in model.parameters()]) # Printing the number of parameters
print(num_params, ' parameters')

load_ckpt = torch.load('best_ckpt_student_distil.pth')
print(load_ckpt['best_epoch'], load_ckpt['accuracy'])
model.load_state_dict(load_ckpt['model'])
model.eval()

prediction_data = {'ID':[], 'Labels':[]}
for i in range(len(test_data)):
    img = test_data[i].reshape(3,32,32).transpose(1,2,0)
    img_tensor = img_transform(img).unsqueeze(0).cuda()
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs.data, 1)
        prediction_data['Labels'].append(predicted.item())
        prediction_data['ID'].append(test_ids[i])

df = pd.DataFrame(data=prediction_data)
df.set_index('ID', inplace=True)
df.to_csv('submission_distil.csv')