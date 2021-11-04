import argparse
import torch
import torch.nn as nn
import torch.utils.data as data
from models import WSDAN
from torch.autograd import Variable
from utils import batch_augment
from evalLoader import BirdLoader


def load_classes(filename):
    classes = []
    fin = open(filename)
    lines = fin.read().splitlines()
    for line in lines:
        classes.append(line)
    return classes


def write_ans(preds, filenames, file):
    preds = torch.argmax(preds, dim=1)
    for i in range(len(preds)):
        file.write(filenames[i] + " " + classes[preds[i].item()] + "\n")


def eval():
    file = open('answer.txt', 'w')
    model.eval()
    with torch.no_grad():
        for step, (imgs, filenames) in enumerate(dataLoader):
            imgs = Variable(imgs).to(args.device)
            # Raw Image
            pred_raw, _,  _, attention_map = model(imgs)
            # Object Localization and Refinement
            crop_images = batch_augment(imgs, attention_map, mode='crop', theta=0.1, padding_ratio=0.05)
            pred_crop, _, _, _ = model(crop_images)
            # Final prediction
            preds = (pred_raw + pred_crop) / 2.
            write_ans(preds, filenames, file)


def set_model():
    model = WSDAN(M=args.num_attentions, net=args.net, pretrained=True)
    model = nn.DataParallel(model).to(args.device)
    model.load_state_dict(torch.load('./weight/resnet152_2'))
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="None")
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_attentions', default=32, type=int)
    parser.add_argument('--net', default='resnet152', type=str)
    args = parser.parse_args()
    print('Single Model Inference')

    classes = load_classes("./data/classes.txt")
    dataset = BirdLoader(root="./data/", img_path="./data/testing_images/")
    dataLoader = data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    model = set_model()
    print("Number of testing set: {}".format(len(dataset)))
    eval()
