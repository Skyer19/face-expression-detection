from PIL import Image
from torch.utils.data import Dataset
import os
import random
import cv2
import numpy as np

class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None): # 主要是数据的获取，比如从某个文件中获取
        fh = open(txt_path, 'r')
        imgs = []

        for line in fh:
            line = line.rstrip()
            words = line.split()

            imgs.append((words[0], int(words[1])))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index): # 读取数据，对数据进行处理
        fn1, label = self.imgs[index]
        img = Image.open(fn1).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self): # 得到数据的长度
        return len(self.imgs)
