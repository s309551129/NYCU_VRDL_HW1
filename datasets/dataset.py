import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as trns
from PIL import Image


class Dataset(data.Dataset):
    def __init__(self, root, img_path, img_size, mode):
        self.root = root
        self.img_path = img_path
        self.mode = mode
        self.dataset = self.load_dataset()
        if self.mode == "train":
            self.data_trns = trns.Compose([
                trns.RandomResizedCrop(img_size),
                trns.RandomHorizontalFlip(p=0.5),
                trns.RandomRotation(30),
                trns.ColorJitter(brightness=0.126, saturation=0.5),
                trns.ToTensor(),
                trns.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.data_trns = trns.Compose([
                trns.Resize([img_size, img_size]),
                trns.ToTensor(),
                trns.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def load_dataset(self):
        dataset = []
        if self.mode == "train":
            fin = open(self.root+"training_labels.txt")
        else:
            fin = open(self.root+"validation_labels.txt")
        lines = fin.read().splitlines()
        for line in lines:
            line = line.split(' ')
            dataset.append((line[0], int(line[1])))
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        filename, label_idx = self.dataset[idx]
        img = Image.open(self.img_path+filename)
        img = self.data_trns(img)
        return img, label_idx
