import os
import torch

from torch import nn
from torch.optim import Adam
from datetime import datetime
from os.path import join, exists
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from rich.progress import track as tk

import del_sort.AFLink.config as cfg

from del_sort.AFLink.dataset import LinkData
from del_sort.AFLink.model import PostLinker


def train(save: bool):
    model = PostLinker()
    model.cuda()
    model.train()
    dataset = LinkData(cfg.root_train, 'train')
    dataloader = DataLoader(
        dataset=dataset, batch_size=cfg.train_batch, shuffle=True, num_workers=cfg.num_workers, drop_last=True)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=cfg.train_lr, weight_decay=cfg.train_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.train_epoch, eta_min=1e-5)
    # validate(model, loss_fn)
    print('======================= Start Training =======================')
    print('======================= AFLinkTraining =======================')
    for epoch in range(cfg.train_epoch):
        print('epoch: %d with lr=%.0e' % (epoch + 1, optimizer.param_groups[0]['lr']))
        loss_sum = 0
        for i, (pair1, pair2, pair3, pair4, label) in tk(enumerate(dataloader), description='epoch:' + str(epoch + 1)):
            optimizer.zero_grad()
            pairs_1 = torch.cat((pair1[0], pair2[0], pair3[0], pair4[0]), dim=0).cuda()
            pairs_2 = torch.cat((pair1[1], pair2[1], pair3[1], pair4[1]), dim=0).cuda()
            label = torch.cat(label, dim=0).cuda()
            output = model(pairs_1, pairs_2)
            loss = loss_fn(output, label)
            loss_sum += loss.item()
            loss.backward()
            optimizer.step()
        scheduler.step()
        print('  loss_train: {:.2f}'.format(loss_sum / len(dataloader)))
        # validate(model)
    if save:
        if not exists(cfg.model_savedir): os.mkdir(cfg.model_savedir)
        torch.save(model.state_dict(), join(cfg.model_savedir, 'AFLinkModel_1.2.0_epoch{}.pth'.format(epoch + 1)))
    return model


def validate(model):
    model.eval()
    dataset = LinkData(cfg.root_train, 'val')
    dataloader = DataLoader(
        dataset=dataset, batch_size=cfg.val_batch, shuffle=False, num_workers=cfg.num_workers, drop_last=False)
    labels = list()
    outputs = list()
    for i, (pair1, pair2, pair3, pair4, label) in enumerate(dataloader):
        pairs_1 = torch.cat((pair1[0], pair2[0], pair3[0], pair4[0]), dim=0).cuda()
        pairs_2 = torch.cat((pair1[1], pair2[1], pair3[1], pair4[1]), dim=0).cuda()
        label = torch.cat(label, dim=0).cuda()
        output = model(pairs_1, pairs_2)
        labels.extend(label.tolist())
        outputs.extend(output.tolist())
    outputs = [0 if x[0] > x[1] else 1 for x in outputs]
    precision = precision_score(labels, outputs, average='macro', zero_division=0)
    recall = recall_score(labels, outputs, average='macro', zero_division=0)
    f1 = f1_score(labels, outputs, average='macro', zero_division=0)
    confusion = confusion_matrix(labels, outputs)
    print('  f1/p/r: {:.2f}/{:.2f}/{:.2f}'.format(f1, precision, recall))
    print('  ConfMat: ', confusion.tolist())
    model.train()


if __name__ == '__main__':
    startTime = datetime.now()
    print(startTime)
    if not cfg.save_model:
        print('====================== Model not saved! ======================')
    train(cfg.save_model)
    if not cfg.save_model:
        print('====================== Model not saved! ======================')
    endTime = datetime.now()
    print(endTime)
    print('Running time: ', (endTime - startTime))
