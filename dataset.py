import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch


class tiny_caltech35(Dataset):
    def __init__(self, transform=None, used_data=['train']):
        self.train_dir = 'dataset/train/'
        self.addition_dir = 'dataset/addition/'
        self.val_dir = 'dataset/val/'
        self.test_dir = 'dataset/test/'
        self.used_data = used_data
        for x in used_data:
            assert x in ['train', 'addition', 'val', 'test']
        self.transform = transform

        self.samples, self.annotions = self._load_samples()

    def _load_samples_one_dir(self, dir='dataset/train/'):
        samples, annotions = [], []
        if 'test' not in dir:
            sub_dir = os.listdir(dir)
            for i in sub_dir:
                tmp = os.listdir(os.path.join(dir, i))
                samples += [os.path.join(dir, i, x) for x in tmp]
                annotions += [int(i)] * len(tmp)
        else:
            with open(os.path.join(self.test_dir, 'annotions.txt'), 'r') as f:
                tmp = f.readlines()
            for i in tmp:
                path, label = i.split(',')[0], i.split(',')[1]
                samples.append(os.path.join(self.test_dir, path))
                annotions.append(int(label))
        return samples, annotions

    def _load_samples(self):
        samples, annotions = [], []
        for i in self.used_data:
            if i == 'train':
                tmp_s, tmp_a = self._load_samples_one_dir(dir=self.train_dir)
            elif i == 'addition':
                tmp_s, tmp_a = self._load_samples_one_dir(dir=self.addition_dir)
            elif i == 'val':
                tmp_s, tmp_a = self._load_samples_one_dir(dir=self.val_dir)
            elif i == 'test':
                tmp_s, tmp_a = self._load_samples_one_dir(dir=self.test_dir)
            else:
                print('error used_data!!')
                exit(0)
            samples += tmp_s
            annotions += tmp_a
        return samples, annotions

    def __getitem__(self, index):
        img_path, img_label = self.samples[index], self.annotions[index]
        img = self._loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, img_label

    def _loader(self, img_path):
        return Image.open(img_path).convert('RGB')

    def __len__(self):
        return len(self.samples)
