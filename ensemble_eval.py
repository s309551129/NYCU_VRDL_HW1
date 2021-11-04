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
    inception_mixed.eval()
    resnet50.eval()
    resnet101.eval()
    resnet152.eval()
    resnet152_2.eval()
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for step, (X, filenames) in enumerate(dataLoader):
            X = X.to(args.device)
            pred_raw_1, _, _, attention_map_1 = inception_mixed(X)
            crop_images_1 = batch_augment(X, attention_map_1, mode='crop', theta=0.1, padding_ratio=0.05)
            pred_crop_1, _, _, _ = inception_mixed(crop_images_1)
            pred_1 = (pred_raw_1 + pred_crop_1) / 2.

            pred_raw_2, _, _, attention_map_2 = resnet50(X)
            crop_images_2 = batch_augment(X, attention_map_2, mode='crop', theta=0.1, padding_ratio=0.05)
            pred_crop_2, _, _, _ = resnet50(crop_images_2)
            pred_2 = (pred_raw_2 + pred_crop_2) / 2.

            pred_raw_3, _, _, attention_map_3 = resnet101(X)
            crop_images_3 = batch_augment(X, attention_map_3, mode='crop', theta=0.1, padding_ratio=0.05)
            pred_crop_3, _, _, _ = resnet101(crop_images_3)
            pred_3 = (pred_raw_3 + pred_crop_3) / 2.

            pred_raw_4, _, _, attention_map_4 = resnet152(X)
            crop_images_4 = batch_augment(X, attention_map_4, mode='crop', theta=0.1, padding_ratio=0.05)
            pred_crop_4, _, _, _ = resnet152(crop_images_4)
            pred_4 = (pred_raw_4 + pred_crop_4) / 2.

            pred_raw_5, _, _, attention_map_5 = resnet152_2(X)
            crop_images_5 = batch_augment(X, attention_map_5, mode='crop', theta=0.1, padding_ratio=0.05)
            pred_crop_5, _, _, _ = resnet152_2(crop_images_5)
            pred_5 = (pred_raw_5 + pred_crop_5) / 2.

            pred = 0.7709*softmax(pred_1) + 0.7511*softmax(pred_2) + 0.7821*softmax(pred_3) + 0.7910*softmax(pred_4) + 0.8058*softmax(pred_5)
            write_ans(pred, filenames, file)


def set_model(model_name, weight_name):
    model = WSDAN(M=args.num_attentions, net=model_name, pretrained=True)
    model = nn.DataParallel(model).to(args.device)
    model.load_state_dict(torch.load('./weight/'+weight_name))
    for param in model.parameters():
        param.requires_grad = False
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="None")
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_attentions', default=32, type=int)
    args = parser.parse_args()
    print('Ensemble Model Inference')
    classes = load_classes("./data/classes.txt")
    dataset = BirdLoader(root="./data/", img_path="./data/testing_images/")
    dataLoader = data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    inception_mixed = set_model('inception_mixed_6e', 'inception_mixed_6e')
    resnet50 = set_model('resnet50', 'resnet50')
    resnet101 = set_model('resnet101', 'resnet101')
    resnet152 = set_model('resnet152', 'resnet152')
    resnet152_2 = set_model('resnet152', 'resnet152_2')
    print("Number of testing set: {}".format(len(dataset)))
    eval()
