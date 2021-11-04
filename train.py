import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models import WSDAN
from datasets import trainDataset, Dataset
from utils import CenterLoss, AverageMeter, TopKAccuracyMetric, batch_augment
torch.backends.cudnn.benchmark = True


def collect_data():
    # Load dataset
    #train_dataset = trainDataset(root="./data/", img_path="./data/training_images/")
    train_dataset = Dataset(root="./data/split_images/", img_path="./data/split_images/training_images/", img_size=args.img_size, mode="train")
    validate_dataset = Dataset(root="./data/split_images/", img_path="./data/split_images/validation_images/", img_size=args.img_size, mode="validate")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    validate_loader = DataLoader(validate_dataset, batch_size=args.batch_size*4, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, validate_loader


def train(**kwargs):
    # Retrieve training configuration
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    feature_center = kwargs['feature_center']
    optimizer = kwargs['optimizer']
    pbar = kwargs['pbar']
    # metrics initialization
    loss_container.reset()
    raw_metric.reset()
    crop_metric.reset()
    drop_metric.reset()
    # begin training
    net.train()
    for i, (X, y) in enumerate(data_loader):
        # obtain data for training
        X, y = X.to(args.device), y.to(args.device)
        optimizer.zero_grad()
        # raw images forward
        y_pred_raw, y_pred_aux, feature_matrix, attention_map = net(X)
        # Update Feature Center
        feature_center_batch = F.normalize(feature_center[y], dim=-1)
        feature_center[y] += args.beta * (feature_matrix.detach() - feature_center_batch)
        # Attention Cropping
        with torch.no_grad():
            crop_images = batch_augment(X, attention_map[:, :1, :, :], mode='crop', theta=(0.4, 0.6), padding_ratio=0.1)
            drop_images = batch_augment(X, attention_map[:, 1:, :, :], mode='drop', theta=(0.2, 0.5))
        aug_images = torch.cat([crop_images, drop_images], dim=0)
        y_aug = torch.cat([y, y], dim=0)
        # crop images forward
        y_pred_aug, y_pred_aux_aug, _, _ = net(aug_images)
        y_pred_aux = torch.cat([y_pred_aux, y_pred_aux_aug], dim=0)
        y_aux = torch.cat([y, y_aug], dim=0)
        # loss
        batch_loss = cross_entropy_loss(y_pred_raw, y) / 3. + \
                     cross_entropy_loss(y_pred_aux, y_aux) * 3. / 3. + \
                     cross_entropy_loss(y_pred_aug, y_aug) * 2. / 3. + \
                     center_loss(feature_matrix, feature_center_batch)
        # backward
        batch_loss.backward()
        optimizer.step()
        # metrics: loss and top-1,5 error
        with torch.no_grad():
            epoch_loss = loss_container(batch_loss.item())
            epoch_raw_acc = raw_metric(y_pred_raw, y)
            epoch_crop_acc = crop_metric(y_pred_aug, y_aug)
            epoch_drop_acc = drop_metric(y_pred_aux, y_aux)
        # end of this batch
        batch_info = 'Loss {:.4f}, Raw Acc ({:.2f}, {:.2f}), Crop Acc ({:.2f}, {:.2f}), Drop Acc ({:.2f}, {:.2f})'.format(
            epoch_loss, epoch_raw_acc[0], epoch_raw_acc[1],
            epoch_crop_acc[0], epoch_crop_acc[1], epoch_drop_acc[0], epoch_drop_acc[1])
        pbar.update()
        pbar.set_postfix_str(batch_info)

    return batch_info


def validate(**kwargs):
    # Retrieve training configuration
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    pbar = kwargs['pbar']
    train_info = kwargs['train_info']
    # metrics initialization
    loss_container.reset()
    raw_metric.reset()
    # begin validation
    net.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(data_loader):
            # obtain data
            X, y = X.to(args.device), y.to(args.device)
            # Raw Image
            y_pred_raw, y_pred_aux, _, attention_map = net(X)
            # Object Localization and Refinement
            crop_images = batch_augment(X, attention_map, mode='crop', theta=0.1, padding_ratio=0.05)
            y_pred_crop, y_pred_aux_crop, _, _ = net(crop_images)
            # Final prediction
            y_pred = (y_pred_raw + y_pred_crop) / 2.
            y_pred_aux = (y_pred_aux + y_pred_aux_crop) / 2.
            # loss
            batch_loss = cross_entropy_loss(y_pred, y)
            epoch_loss = loss_container(batch_loss.item())
            # metrics: top-1,5 error
            epoch_acc = raw_metric(y_pred, y)
            aux_acc = drop_metric(y_pred_aux, y)

    batch_info = 'Val Loss {:.4f}, Val Acc ({:.2f}, {:.2f})'.format(epoch_loss, epoch_acc[0], epoch_acc[1])
    pbar.set_postfix_str('{}, {}'.format(train_info, batch_info))
    return epoch_acc[0]


def run():
    best_valid_acc = 0.0
    for epoch in range(args.epoch):
        pbar = tqdm(total=len(train_loader), unit=' batches')
        pbar.set_description('Epoch {}/{}'.format(epoch + 1, args.epoch))
        train_info = train(data_loader=train_loader, net=net, feature_center=feature_center, optimizer=optimizer, pbar=pbar)
        valid_acc = validate(data_loader=validate_loader, net=net, pbar=pbar, train_info=train_info)
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            #torch.save(net.state_dict(), './weight/'+args.net)
        scheduler.step()
        pbar.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="None")
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('--epoch', default=160, type=int)
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--num_attentions', default=32, type=int)
    parser.add_argument('--beta', default=5e-2, type=float)
    parser.add_argument('--img_size', default=448, type=int)
    parser.add_argument('--net', default='resnet152', type=str)
    args = parser.parse_args()
    print(args.net)
    train_loader, validate_loader = collect_data()
    # Initialize model
    net = WSDAN(M=args.num_attentions, net=args.net, pretrained=True)
    # feature_center: size of (#classes, #attention_maps * #channel_features)
    feature_center = torch.zeros(200, args.num_attentions * net.num_features).to(args.device)
    net = nn.DataParallel(net).to(args.device)
    # Optimizer, LR Scheduler
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
    # General loss functions
    cross_entropy_loss = nn.CrossEntropyLoss()
    center_loss = CenterLoss()
    # loss and metric
    loss_container = AverageMeter(name='loss')
    raw_metric = TopKAccuracyMetric(topk=(1, 5))
    crop_metric = TopKAccuracyMetric(topk=(1, 5))
    drop_metric = TopKAccuracyMetric(topk=(1, 5))
    run()
