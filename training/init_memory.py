import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from training.train_utils import AverageMeter, ProgressMeter, accuracy

import warnings
warnings.filterwarnings("ignore")
warnings.warn("deprecated", DeprecationWarning)

def init_memory(train_loader, model,Memory_Bank, Memory_Bank_upper, criterion,
                                optimizer,epoch, args):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [losses, top1, top5],
        prefix="Init Epoch: [{}]".format(epoch))
    # switch to train mode
    # model在main_worker的第111行
    model.train()
    # for i, (images, _) in enumerate(train_loader):
    for i, (data0, label0, data1, label1) in enumerate(train_loader):
        # measure data loading time
        # 通道置乱增强
        # batchsize, channels, height, width = data1.size()
        # groups = 10
        # channels_per_group = int(channels /groups)
        # x = data1.view(batchsize, groups, channels_per_group, height, width)
        # x = x.transpose(1, 2).contiguous()
        # x = x.view(batchsize, -1, height, width)
        # data1 = x

        # 通道分组增强
        # data0 = data0[:,:100,:,:]
        # data1 = data1[:,100:,:,:]
        # images = [data0, data1]

        # mixup增强
        # alpha = 0.5  # 默认设置为1
        # lam = np.random.beta(alpha, alpha)
        # index = torch.randperm(data0.size(0))
        # images_a, images_b = data0, data0[index]
        # labels_a, labels_b = label0, label0[index]
        # data1 = lam * images_a + (1 - lam) * images_b
        # outputs = model(mixed_images)
        # _, preds = torch.max(outputs, 1)
        # loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        
        # 高斯模糊增强
        # data0_std = torch.from_numpy(np.random.normal(1, 0.025, size=data0.size())).float()
        # data0 = data0 * data0_std
        # data1_std = torch.from_numpy(np.random.normal(1, 0.075, size=data0.size())).float()
        # data1 = data1 * data1_std
        images = [data0, data1]
        
        if args.dataset=='IndianPines' or args.dataset=='PaviaU' or args.dataset=='HyRANK' \
            or args.dataset=='Botswana' or args.dataset=='Houston' or args.dataset=='Houston18' \
                or args.dataset=='Chikusei':
            if args.gpu is not None:
                for k in range(len(images)):  
                    images[k] = images[k].cuda(args.gpu, non_blocking=True)
        # cifar10
        elif args.dataset=='cifar10':
            if args.gpu is not None:
                for k in range(len(images)): 
                    print("打印k",k)
                    images[k] = images[k].cuda(args.gpu, non_blocking=True)
                    print("打印images[k].shape",images[k].shape)  # torch.Size([32, 3, 32, 32])
        # imagenet-mini
        elif args.dataset=='imagenet-mini':
            if args.gpu is not None:
                for k in range(len(images)): 
                    print("打印k",k)
                    images[k] = images[k].cuda(args.gpu, non_blocking=True)
                    print("打印images[k].shape",images[k].shape)  # torch.Size([32, 3, 224, 224])
        
        # compute output   
        # q, _, _, k  = model(im_q=images[0], im_k=images[1])
        q, _, _, k, q_upper, _, _, k_upper  = model(im_q=images[0], im_k=images[1])
        q_upper = q_upper.view(q_upper.size(0), -1)
        k_upper = k_upper.view(k_upper.size(0), -1)

        
        # 添加上层小字典
        _, _, l_neg_upper = Memory_Bank_upper(q_upper)
        l_pos_upper = torch.einsum('nc,nc->n', [q_upper, k_upper]).unsqueeze(-1)
        logits_upper = torch.cat([l_pos_upper, l_neg_upper], dim=1)
        logits_upper /= 0.2
        labels_upper = torch.zeros(logits_upper.shape[0], dtype=torch.long).cuda(args.gpu)
        




        # print("q.shape, k.shape", q.shape, k.shape)
        # print("q,k初始化完毕")
        d_norm, d, l_neg = Memory_Bank(q)
        # print("l_neg.shape: ", l_neg.shape)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # print("l_pos.shape: ", l_pos.shape)
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        # print("logits.shape: ", logits.shape)
        # print(logits[0][0])
        logits /= 0.2#using the default param in MoCo temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda(args.gpu)
        loss = criterion(logits, labels) + criterion(logits_upper, labels_upper)
        # loss = criterion(q, k)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 这里也有一个准确率
        acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
        
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1.item(), images[0].size(0))
        top5.update(acc5.item(), images[0].size(0))

        if i % args.print_freq == 0 and args.rank==0:
            progress.display(i)

        output = k
        # output = concat_all_gather(k)
       
        batch_size = output.size(0)
        start_point = i * batch_size
        end_point = min((i + 1) * batch_size, args.cluster)
        Memory_Bank.W.data[:, start_point:end_point] = output[:end_point - start_point].T
        

        # 添加字典更新
        output_upper = k_upper
        batch_size_upper = output_upper.size(0) #// 2  # 如果除以2了会掉0.5个点
        start_point_upper = i * batch_size_upper
        end_point_upper = min((i + 1) * batch_size_upper, args.cluster_upper)
        # print(batch_size, batch_size_upper, "batch_size, batch_size_upper")
        # print(start_point, end_point, start_point_upper, end_point_upper, "start and end point")
        Memory_Bank_upper.W.data[:, start_point_upper:end_point_upper] = k_upper[:end_point_upper - start_point_upper].T
        

        if (i+1) * batch_size >= args.cluster:
            break
    #if args.nodes_num>1:
    #    for param_q, param_k in zip(model.encoder_q.parameters(),
    #                            model.encoder_k.parameters()):
    #        param_k.data.copy_(param_q.data)  # initialize
    #else:
    for param_q, param_k in zip(model.encoder_q.parameters(),
                                model.encoder_k.parameters()):
        param_k.data.copy_(param_q.data)  # initialize
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output