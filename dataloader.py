import os
from glob import glob

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def fetch_image()


class Cifar10Dataset(Dataset):
    def __init__(self, root_dir='dataset/cifar-10-python/cifar-10-batches-py', split='train'):
        self.train_file_names = ['data_batch_1, data_batch_2, data_batch_3, data_batch_4, data_batch_5']
        self.test_file_name = 'test_batch'
        self.split = split
        self.root_dir = root_dir

        if self.split != 'train' and self.split != 'test':
            print("Invalid split mentioned. It should be either 'train' or 'test'.")
            exit()
        
    def __getitem__(self, index):
        file_index = index // 10000
        image_index = index % 10000
        if self.split == 'train':
            file_name = os.path.join(root_dir, )
        