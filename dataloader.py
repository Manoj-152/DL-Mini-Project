import os
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def fetch_image(dict, index):
    pic = dict[b'data'][index].reshape(3,32,32).transpose(1,2,0)
    label = dict[b'labels'][index]
    return pic, label


class Cifar10Dataset(Dataset):
    def __init__(self, root_dir='dataset/cifar-10-python/cifar-10-batches-py', split='train'):
        self.train_file_names = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
        self.test_file_name = 'test_batch'
        self.split = split
        self.root_dir = root_dir

        if self.split != 'train' and self.split != 'test':
            print("Invalid split mentioned. It should be either 'train' or 'test'.")
            exit()
        self._init_transform()


    def _init_transform(self):
        if self.split == 'train':
            self.img_transform = transforms.Compose([
                        transforms.ToTensor(),
                        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ])
        elif self.split == 'test':
            self.img_transform =  transforms.Compose([
                        transforms.ToTensor(),
                        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ])


    def __getitem__(self, index):
        file_index = index // 10000
        image_index = index % 10000
        if self.split == 'train':
            file_name = os.path.join(self.root_dir, self.train_file_names[file_index])
        elif self.split == 'test':
            file_name = os.path.join(self.root_dir, self.test_file_name)

        images_dict = unpickle(file_name)
        pic, label = fetch_image(images_dict, image_index)
        pic = self.img_transform(pic)
        label = torch.Tensor([label])

        return pic, label

    def __len__(self):
        if self.split == 'train':
            total_len = 0
            for path in self.train_file_names:
                file_path = os.path.join(self.root_dir, path)
                images_dict = unpickle(file_path)
                total_len += len(images_dict[b'labels'])
        
        elif self.split == 'test':
            file_path = os.path.join(self.root_dir, self.test_file_name)
            images_dict = unpickle(file_path)
            total_len = len(images_dict[b'labels'])

        return total_len


if __name__ == '__main__':
    dataset = Cifar10Dataset(split='train')
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    pic, label = next(iter(dataloader))
    print("Input Shape:", pic.shape)
    print("Target Shape:", label.shape)
    plt.imshow(pic[0].permute(1,2,0))
    plt.savefig('scrap.png')
    plt.show()
    print(label[0])