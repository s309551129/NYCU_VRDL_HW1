import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as trns
from PIL import Image


class BirdLoader(data.Dataset):
    def __init__(self, root, img_path):
        self.root = root
        self.img_path = img_path
        self.img_order = self.load_img_order()
        self.data_trns = trns.Compose([
            trns.Resize([448, 448]),
            trns.ToTensor(),
            trns.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def load_img_order(self):
        img_order = []
        fin = open(self.root+"testing_img_order.txt")
        lines = fin.read().splitlines()
        for line in lines:
            img_order.append(line)
        return img_order

    def __len__(self):
        return len(self.img_order)

    def __getitem__(self, idx):
        filename = self.img_order[idx]
        img = Image.open(self.img_path+filename)
        img = self.data_trns(img)
        return img, filename
