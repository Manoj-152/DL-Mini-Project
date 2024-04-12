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

parser = argparse.ArgumentParser(description="Test Dataset Evaluation")
parser.add_argument('--model_ckpt', help='mention the model checkpoint')
parser.add_argument('--dataset_path', default='data/cifar_test_nolabels.pkl', help='mention the test dataset path')
args = parser.parse_args()

# Reading the test dataset
test_dict = unpickle(args.dataset_path)
test_data, test_ids = test_dict[b'data'], test_dict[b'ids']

img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

# Creating the model and loading the pretrained weights
model = Resnet(BasicBlock, [2,1,1,1], 10)
model = model.cuda()
num_params = np.sum([p.nelement() for p in model.parameters()]) # Printing the number of parameters
print(num_params, ' parameters')

load_ckpt = torch.load(args.model_ckpt)
print("Validation accuracy obtained previously:", load_ckpt['accuracy'])
model.load_state_dict(load_ckpt['model'])
model.eval()

# Now we are passing in the test data and noting down the predictions
prediction_data = {'ID':[], 'Labels':[]}
for i in range(len(test_data)):
    img = test_data[i].reshape(3,32,32).transpose(1,2,0)
    img_tensor = img_transform(img).unsqueeze(0).cuda()
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs.data, 1)
        prediction_data['Labels'].append(predicted.item())
        prediction_data['ID'].append(test_ids[i])

# Saving the predictions in the csv file
df = pd.DataFrame(data=prediction_data)
df.set_index('ID', inplace=True)
df.to_csv('submission.csv')