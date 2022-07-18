from random import random
from sched import scheduler
import numpy as np
from tkinter.ttk import LabeledScale
from matplotlib.pyplot import step
from torchvision import models
from torch.utils.data import DataLoader
from dataset import Tissue
from utils import (get_ids_and_label_db, 
                   split_train_valid,
                   dict2obj)
from apex import amp
from ruamel import yaml
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import wandb
import time
import argparse
import random
import torch.optim.lr_scheduler as lr_scheduler
import math
from model import NephNet2D, ResNet


def train_epoch(model: nn.Module, epoch: int, dataloader: DataLoader, optimizer, device, conf):
    model.train()
    pred_true = 0
    pred_tot = 0
    loss_sum = 0
    log_image = None
    log_label = None

    # register hook, depend on model arch
    # watch_layer = ['conv1', 'layer2', 'layer3', 'layer4']
    watch_layer = []
    feature_map = {}
    def get_feature_map(name):
        def hook(model: nn.Module, input, output: torch.Tensor):
            feature_map[name] = output.detach()
        return hook
    for name, child in model.named_children():
        if name in watch_layer:
            child.register_forward_hook(get_feature_map(name))

    # train batch
    for iter, batch in enumerate(dataloader):
        image, label = batch
        log_image = image.detach()
        log_label = label.detach()
        image = image.to(torch.device(device))
        label = label.to(torch.device(device))
        predict = model.forward(image)

        regularization_loss = 0
        for param in model.parameters():
            regularization_loss += torch.sum(abs(param))

        loss = F.cross_entropy(predict, label)
        loss += regularization_loss * conf.exp.l1_norm_weight

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, pred_class = predict.max(1)
        pred_true += (label == pred_class).sum()
        pred_tot += label.size(0)
        loss_sum += loss.item()

        batch_acc = (label == pred_class).sum() / label.size(0)
        # print("[Train:{}-{}/{}]\tAcc:{}\tLoss:{}".format( 
        #       epoch, 
        #       iter, 
        #       len(dataloader), 
        #       batch_acc, 
        #       loss))
    
    logger_visualize = []

    for i in range(3):
        logger_visualize.append(
            # wandb.Image((log_image[i] * 57.59275746686797 + 57.59275746686797), 
            wandb.Image(log_image[i], 
                         caption=f"origin image, label={log_label[i]}")
        )
        for k, v in feature_map.items():
            # print(v.size())
            logger_visualize.append(
                # wandb.Image(v[i].mean(dim=0) * 57.59275746686797 + 57.59275746686797,
                wandb.Image(v[i].mean(dim=0),
                            caption=f"feature map {k}")
            ) 

    epoch_acc = pred_true / pred_tot
    epoch_loss = loss_sum / len(dataloader)
    return epoch_acc, epoch_loss, logger_visualize


@torch.no_grad()
def evaluate(model: nn.Module, epoch: int, dataloader: DataLoader, device, conf):
    model.eval()
    pred_true = 0
    pred_tot = 0
    loss_sum = 0

    for iter, batch in enumerate(dataloader):
        image, label = batch
        image = image.to(torch.device(device))
        label = label.to(torch.device(device))
        predict = model.forward(image)

        regularization_loss = 0
        for param in model.parameters():
            regularization_loss += torch.sum(abs(param))

        loss = F.cross_entropy(predict, label)
        loss += regularization_loss * conf.exp.l1_norm_weight

        _, pred_class = predict.max(1)
        pred_true += (label == pred_class).sum()
        pred_tot += label.size(0)
        loss_sum += loss.item()

        batch_acc = (label == pred_class).sum() / label.size(0)
        # print("[Valid:{}-{}/{}]\tAcc:{}\tLoss:{}".format(
        #       epoch, 
        #       iter, 
        #       len(dataloader), 
        #       batch_acc, 
        #       loss))

    epoch_acc = pred_true / pred_tot
    epoch_loss = loss_sum / len(dataloader)
    return epoch_acc, epoch_loss 
    

def main(conf):

    print("INFO: load label and ids")
    ids, label_db = get_ids_and_label_db(
        conf.dataset.label_csv_path)
    train_ids, valid_ids = split_train_valid(ids)

    print("INFO: load data")
    dataset_train, dataset_valid = \
        Tissue(train_ids, label_db, conf.dataset.image_root), \
        Tissue(valid_ids, label_db, conf.dataset.image_root)
    dataloader_train, dataloader_valid = \
        DataLoader(dataset_train, 
                   conf.exp.train_batch_size, 
                   shuffle=True), \
        DataLoader(dataset_valid, 
                   conf.exp.valid_batch_size, 
                   shuffle=True), 

    print("INFO: load model")
    # model = models.resnet18()
    # hidden_feature_size = model.fc.in_features
    # model.fc = nn.Linear(hidden_feature_size, conf.model.out_dim)
    model = NephNet2D(conf.model.num_feature)
    # model = ResNet()

    model = model.to(conf.env.device)

    # optimizer = optim.Adam(
    #     model.parameters(),
    #     lr=conf.exp.learning_rate,
    #     weight_decay=conf.exp.weight_decay
    # )
    optimizer = optim.SGD(
        model.parameters(),
        lr=conf.exp.learning_rate,
        weight_decay=conf.exp.weight_decay,
        momentum=0.8,
        nesterov=True
    )
    # apex
    model, optimizer = amp.initialize(model, optimizer, opt_level='O1') 

    wandb.watch(model, log="all", log_freq=1)

    # warm_up_with_cosine_lr = lambda epoch: epoch / 5 if epoch <= 5 else 0.5 * (math.cos((epoch - 5) / (conf.exp.epoch_num - 5) * math.pi) + 1)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    # warm_up = lambda epoch: conf.exp.learning_rate * 0.9 / 5 * epoch \
    #                         + conf.exp.learning_rate * 0.1 if epoch <= 5 else \
    #                         conf.exp.learning_rate
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 
    #                                            conf.exp.learning_rate, 
    #                                            conf.exp.learning_rate / 10)
    # scheduler = lr_scheduler.LinearLR(
    #     optimizer,
    #     conf.exp.learning_rate,
    #     conf.exp.learning_rate * 0.1,
    #     conf.exp.epoch_num
    # )
    # scheduler = lr_scheduler.CosineAnnealingLR(
    #                             optimizer, 
    #                             10, 
    #                             eta_min=0, 
    #                             last_epoch=-1, 
    #                             verbose=False
    # )
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        factor=0.29
    )
    for epoch in range(conf.exp.epoch_num):
        train_acc, train_loss, logger_visualize = train_epoch(model, 
                                            epoch, 
                                            dataloader_train, 
                                            optimizer, 
                                            conf.env.device,
                                            conf)
        valid_acc, valid_loss = evaluate(model, 
                                         epoch, 
                                         dataloader_valid, 
                                         conf.env.device,
                                         conf)
        scheduler.step(valid_loss)
        # scheduler.step()
        # print('*' * 20)
        print("[Train:{}]\tAcc:{}\tLoss:{}".format(epoch, train_acc, train_loss))
        print("[Valid:{}]\tAcc:{}\tLoss:{}".format(epoch, valid_acc, valid_loss))
        # print('*' * 20)
        wandb.log({
            'train':{
                'Acc': train_acc,
                'Loss': train_loss
            },
            'valid': {
                'Acc': valid_acc,
                'Loss': valid_loss
            },
            'env/learning rate': optimizer.param_groups[0]["lr"],
            'feature_map': logger_visualize
        })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()     
    parser.add_argument('--config')
    args = parser.parse_args()

    confDict = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    conf = dict2obj(confDict)
    wandb.init(
        project='PML',
        name='PML:' + time.asctime(),
        config=confDict,
    )
    torch.manual_seed(conf.env.random_seed)
    np.random.seed(conf.env.random_seed)
    random.seed(conf.env.random_seed)
    main(conf)