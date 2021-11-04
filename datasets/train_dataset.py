import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as trns
from PIL import Image


class trainDataset(data.Dataset):
    def __init__(self, root, img_path):
        self.root = root
        self.img_path = img_path
        self.num_classes = 200
        self.dataset = self.load_dataset()
        self.data_trns = trns.Compose([
            trns.RandomResizedCrop(448),
            trns.RandomHorizontalFlip(p=0.5),
            trns.RandomRotation(30),
            trns.ColorJitter(brightness=0.126, saturation=0.5),
            trns.ToTensor(),
            trns.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def load_dataset(self):
        dataset = []
        fin = open(self.root+"training_labels.txt")
        lines = fin.readlines()
        for line in lines:
            line = line.split(' ')
            filename = line[0]
            idx = int(line[1].split('.')[0]) - 1
            dataset.append((filename, idx))
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        filename, label_idx = self.dataset[idx]
        img = Image.open(self.img_path+filename)
        img = self.data_trns(img)
        return img, label_idx
