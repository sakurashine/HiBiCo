#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import math
import shutil
import os
import torch
import spectral
from sklearn.decomposition import PCA
import numpy as np

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        filename = os.path.abspath(filename)
        root_file_path = os.path.split(filename)[0]
        new_path = os.path.join(root_file_path,'model_best.pth.tar')
        try:
            shutil.copyfile(filename, new_path)
        except:
            return

# 管理变量更新，对平均值和当前值进行计算和存储
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def write(self,log_path,batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        with open(log_path,'a+') as file:
            file.write('\t'.join(entries))
            file.write("\n")
            file.flush()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos==1:  # cosine lr schedule
        lr = args.lr_final + 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (args.lr - args.lr_final)
    elif args.cos==2:
        lr*=math.cos(math.pi * epoch / (args.epochs*2))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate2(optimizer, epoch, args,init_lr):
    """Decay the learning rate based on schedule"""
    lr = init_lr
    ratio = args.lr/args.lr_final
    if epoch < args.warmup_epochs:
        lr = init_lr * (epoch+1) / args.warmup_epochs 
    else:
        lr = init_lr/ratio + 0.5 * (1. + math.cos(math.pi * (epoch-args.warmup_epochs)/ (args.epochs-args.warmup_epochs))) * (init_lr- init_lr/ratio)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy_prev(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
def accuracy(output, target, topk=(1,)):
    """
    :param output: predicted prob vectors
    :param target: ground truth
    :param topk: top k predictions considered
    :return:
    Computes the accuracy over the k top predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = torch.sum(correct[:k])#correct[:k].view(-1).float().sum(0, keepdim=True)
            result=correct_k*100.0 / batch_size
            res.append(result)
        return res