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
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch.nn.functional as F
import copy
from training.dcl_loss import *
from sklearn.metrics.pairwise import cosine_similarity

def train_caco(train_loader, model, Memory_Bank, Memory_Bank_upper, criterion,
          optimizer, epoch, args, train_log_path,moco_momentum):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    mem_losses = AverageMeter('MemLoss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, mem_losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    
    # 更新memory bank的学习率
    if epoch<args.warmup_epochs:
        cur_memory_lr =  args.memory_lr* (epoch+1) / args.warmup_epochs 
    elif args.memory_lr != args.memory_lr_final:
        cur_memory_lr = args.memory_lr_final + 0.5 * \
                   (1. + math.cos(math.pi * (epoch-args.warmup_epochs) / (args.epochs-args.warmup_epochs))) \
                   * (args.memory_lr- args.memory_lr_final)
    else:
        cur_memory_lr = args.memory_lr
    # print("current memory lr %f"%cur_memory_lr)
    cur_adco_t =args.mem_t
    end = time.time()
    
    # 初始化用于保存当前epoch训练好的feature bank和feature labels
    # feature_bank = torch.randn(1, args.mlp_dim).cuda()  # 如果不加.cuda，那么初始化的tensor是在cpu上的，无法与gpu上的tensor进行运算
    # feature_labels = torch.randn(1).cuda()
    # avgpool = nn.AdaptiveAvgPool2d((1, 1))
    # MemaxLoss = []
    # Alpha = []
    Labels = []
    # train_loader的长度应该是地物像素点个数除以batch数，比如batch size取32时，len(train_loader)=529,32*529=16928(此处对16928存疑？)
    for i, (data0, label0, data1, label1) in enumerate(train_loader):
    # for i, (images, target) in enumerate(train_loader):
        # print("打印train_loader.images and target", images[0].shape, images[1].shape, target)
        # data0.shape torch.Size([64, 103, 11, 11])  # 即B,C,H,W
        # label0.shape torch.Size([64])
        
        # 通道置乱增强
        # batchsize, channels, height, width = data1.size()
        # groups = 10
        # channels_per_group = int(channels /groups)
        # x = data1.view(batchsize, groups, channels_per_group, height, width)
        # x = x.transpose(1, 2).contiguous()
        # x = x.view(batchsize, -1, height, width)
        # data1 = x
        # images = [data0, data1]
        
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

        # print("验证data0和data1的数据差异:")
        # print("data0:", data0[0][0])
        # print("data1:", data1[0][0])

        # 高斯模糊增强
        # data0_std = torch.from_numpy(np.random.normal(1, 0.025, size=data0.size())).float()
        # data0 = data0 * data0_std
        # data1_std = torch.from_numpy(np.random.normal(1, 0.075, size=data0.size())).float()
        # data1 = data1 * data1_std
        # lamda = 0.4
        # data0 = data0 * lamda + data1 * (1-lamda)
        # data1 = data1 * lamda + data0 * (1-lamda)
        images = [data0, data1]

        # 通道间隔分组增强
        # data0.shape, data1.shape  # torch.Size([32, 200, 15, 15])
        # batchsize, channels, height, width = data1.size()
        # groups = 100  # 仅针对IndianPines
        # channels_per_group = int(channels /groups)
        # x = data1.view(batchsize, groups, channels_per_group, height, width)
        # x = x.transpose(1, 2).contiguous()
        # x = x.view(batchsize, -1, height, width)
        # datax = x  # [1,3,5,7,9,...,199,2,4,6,8,...,200]
        # data0 = datax[:,:100,:,:]
        # data1 = datax[:,100:,:,:]
        # images = [data0, data1]
        # for i in range(8):
        #     print("data0第"+str(i)+"个通道的值：", data0[0][i])
        # for i in range(8):
        #     print("data1第"+str(i)+"个通道的值：", data1[0][i])


        # target = label0.cuda(non_blocking=True)
        # target = target.cuda(non_blocking=True)
        
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            for k in range(len(images)):
                images[k] = images[k].cuda(args.gpu, non_blocking=True)
        
        batch_size = images[0].size(0)
        if args.multi_crop:
            update_multicrop_network(model, images, args, Memory_Bank, losses, top1, top5,
                               optimizer, criterion, mem_losses, moco_momentum, cur_memory_lr, cur_adco_t)
        else:
            # 取到q encoder出来的的feature
            _, _, _, Dyrloss, alpha, labels1 = update_sym_network(model, images, args, Memory_Bank, Memory_Bank_upper, losses, top1, top5,
            optimizer, criterion, mem_losses,moco_momentum,cur_memory_lr,cur_adco_t)
            Labels.append(labels1)
            # MemaxLoss.append(Dyrloss)
            # Alpha.append(alpha)
        # print("*******************")
        # print(Meloss)
        # print(A)

        # 制作feature_bank和feature_labels
        # feature = avgpool(k_emb)
        # feature = torch.flatten(q_emb, 1)
        # feature = F.normalize(feature, dim=1)
        # feature_bank.append(feature)
        # feature_labels.append(target)
        # feature_bank = torch.cat([feature_bank, q_emb], dim=0)
        # feature_labels = torch.cat([feature_labels, target], dim=0)
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.rank==0:
            progress.display(i)
            if args.rank == 0:
                progress.write(train_log_path, i)
    
    # 取到当前epoch更新完的Memory Bank
    # 待补充
    
    # print("打印feature_bank",feature_bank.shape)       
    # print("feature_labels",feature_labels.shape)                
    return top1.avg, Dyrloss, alpha, Labels


# 更新query分支
def update_sym_network(model, images, args, Memory_Bank, Memory_Bank_upper,
                   losses, top1, top5, optimizer, criterion, mem_losses,
                   moco_momentum,memory_lr,cur_adco_t):
    model.zero_grad()
    # 四个对象均是torch.Size([batch_size, moco_dim])，在CaCo.py的forward_withoutpred_sym方法处返回
    # q_pred和k_pred是image_q和image_k经过encoder_q的结果，q和k是image_q和image_k经过encoder_k的结果
    # q_pred, k_pred, q, k = model(im_q=images[0], im_k=images[1],run_type=0,moco_momentum=moco_momentum)
    q_pred, k_pred, q, k, q_pred_upper, k_pred_upper, q_upper, k_upper = model(im_q=images[0], im_k=images[1],run_type=0,moco_momentum=moco_momentum)
    # print(q_pred.shape, k.shape, "xxxx")
    
    
    # 有什么损失，能直接拉近两个特征之间的距离。这里可以从字典里挑次近的样本作为正样本做一次cutmix构造正样本
    # 怎么从一个batch里找到跟锚点样本最相似的样本
    
    # import numpy as np
    # from sklearn.metrics import mean_squared_error

    # def compute_mse(image1, image2):
    #     # Assuming images are numpy arrays
    #     return mean_squared_error(image1, image2)

    # anchor_image = q_pred[0].cpu()
    # batch_images = q_pred[1:].cpu()
    # print("shape before", anchor_image.shape)
    # print("shape after", anchor_image.shape)

    # # Compute MSE with anchor image for each image in the batch
    # mse_values = [compute_mse(anchor_image, image) for image in batch_images]

    # # Find the index of the image with the lowest MSE (most visually similar)
    # most_similar_index = np.argmin(mse_values)
    # print("第",most_similar_index, "张是最相似的图")

    # # The most similar image is
    # most_similar_image = batch_images[most_similar_index]

    ##
    '''

    
    import numpy as np

    def compute_similarity(anchor_embedding, batch_embeddings, metric='cosine'):
        if metric == 'cosine':
            # Use cosine similarity
            similarities = np.dot(anchor_embedding, batch_embeddings.T) / (np.linalg.norm(anchor_embedding) * np.linalg.norm(batch_embeddings, axis=1))
        elif metric == 'euclidean':
            # Use Euclidean distance (negative to convert distance to similarity)
            similarities = -np.linalg.norm(anchor_embedding - batch_embeddings, axis=1)
        else:
            raise ValueError("Unsupported similarity metric")

        return similarities

    # Example usage
    anchor_image = batch_of_images[0]
    batch_images = batch_of_images[1:]

    # Assuming you have embeddings for the images (you might use a pre-trained model)
    anchor_embedding = compute_embedding(anchor_image)
    batch_embeddings = [compute_embedding(image) for image in batch_images]

    # Compute similarity scores
    similarities = compute_similarity(anchor_embedding, batch_embeddings)

    # Find the most similar image index
    most_similar_index = np.argmax(similarities)

    # The most similar image is
    most_similar_image = batch_images[most_similar_index]
    '''

    # batch相似性对齐
    # q_corrxy = q_pred @ q_pred.T
    # k_corrxy = k @ k.T

    # Corrxy_q = torch.nn.functional.normalize(q_corrxy)
    # Corrxy_k = torch.nn.functional.normalize(k_corrxy)
    # simi_loss = torch.tensor(cosine_similarity(Corrxy_q.reshape(1, -1), Corrxy_k.reshape(1, -1))).cuda()
    # simi_loss = cosine_similarity(Corrxy_q.reshape(1, -1), Corrxy_k.reshape(1, -1))
    

    # def cosineSimi(A, B):
    #     # Assuming A and B are torch tensors
    #     dot_product = torch.sum(A * B)
    #     norm_A = torch.norm(A)
    #     norm_B = torch.norm(B)
    #     # norm_A = A
    #     # norm_B = B
    #     similarity = dot_product / (norm_A * norm_B)
    #     return similarity.item()
    # simi_loss = cosineSimi(q_corrxy, k_corrxy)
    # print("simi_loss: ", simi_loss)

    
    
    # 变分q
    # q_pred = q_pred + torch.randn_like(q_pred) * args.std
    # q_pred_upper = q_pred_upper + torch.randn_like(q_pred_upper) * args.std
    
    # 浅层字典输出logits
    d_norm1_upper, d1_upper, logits1_upper = Memory_Bank_upper(q_pred_upper)
    d_norm2_upper, d2_upper, logits2_upper = Memory_Bank_upper(k_pred_upper)
    with torch.no_grad():
        logits1_upper_keep = logits1_upper.clone()
        logits2_upper_keep = logits2_upper.clone()
    
    logits1_upper /= args.moco_upper_t #固定值0.08
    logits2_upper /= args.moco_upper_t

    # 浅层字典查找正样本
    with torch.no_grad():
        #swap relationship, im_k supervise im_q
        _, _, check_logits1_upper = Memory_Bank_upper(k_upper)
        logits_fix1_upper = copy.deepcopy(check_logits1_upper)
        check_logits1_upper = check_logits1_upper.detach()
        filter_index1_upper = torch.argmax(check_logits1_upper, dim=1)
        labels1_upper = copy.deepcopy(filter_index1_upper)
        filter_findex1_upper = torch.argmin(check_logits1_upper, dim=1)
        flabels1_upper = copy.deepcopy(filter_findex1_upper)

        # print(check_logits1_upper.shape, "check_logits1_upper")
        # print("logits1_upper,logits1_upper,logits1_upper,logits1_upper,logits1_upper")
        # print(check_logits1_upper[0])
        # print(logits1_upper[0])
        # print(labels1_upper, "labels1_upper")

        _, _, check_logits2_upper = Memory_Bank_upper(q_upper)
        logits_fix2_upper = copy.deepcopy(check_logits2_upper)
        check_logits2_upper = check_logits2_upper.detach()
        filter_index2_upper = torch.argmax(check_logits2_upper, dim=1)
        labels2_upper = copy.deepcopy(filter_index2_upper)
    
    
    # logits1和logits2是q_pred和k_pred经过memory bank字典查找出来的结果
    d_norm1, d1, logits1 = Memory_Bank(q_pred)
    d_norm2, d2, logits2 = Memory_Bank(k_pred)
    # 对memory bank的表征做聚类
    # print(q_pred.shape)
    # print(Memory_Bank.W.T[0].shape)
    # print("OKKK")
    #对logits1加扰动 Dy4hot+1.6KL 93.73
    # disturbance_upper = torch.from_numpy(np.random.uniform(0.99, 1.01, size=logits1_upper.size(1))).float().cuda()
    # disturbance = torch.from_numpy(np.random.uniform(0.99, 1.01, size=logits1.size(1))).float().cuda()
    # logits1_upper = logits1_upper * disturbance_upper
    # logits1 = logits1 * disturbance

    # 变分字典
    # logits1_upper_std = torch.from_numpy(np.random.normal(1, 0.025, size=logits1_upper.size(1))).float().cuda()
    # logits1_upper = logits1_upper * logits1_upper_std
    # logits1_std = torch.from_numpy(np.random.normal(1, 0.025, size=logits1.size(1))).float().cuda()
    # logits1 = logits1 * logits1_std
    
    # 对字典选到的正样本加采样，然后算caco_loss
    # logits1_std1 = torch.from_numpy(np.random.normal(1, 0.025, size=logits1.size(1))).float().cuda()
    # logits1_std2 = torch.from_numpy(np.random.normal(1, 0.025, size=logits1.size(1))).float().cuda()
    # logits1_std3 = torch.from_numpy(np.random.normal(1, 0.025, size=logits1.size(1))).float().cuda()
    # logits1_std4 = torch.from_numpy(np.random.normal(1, 0.025, size=logits1.size(1))).float().cuda()
    # logits1a = logits1 * logits1_std1
    # logits1b = logits1 * logits1_std2
    # logits1c = logits1 * logits1_std3
    # logits1d = logits1 * logits1_std4
    # logits2_std1 = torch.from_numpy(np.random.normal(1, 0.025, size=logits2.size(1))).float().cuda()
    # logits2_std2 = torch.from_numpy(np.random.normal(1, 0.025, size=logits2.size(1))).float().cuda()
    # logits2_std3 = torch.from_numpy(np.random.normal(1, 0.025, size=logits2.size(1))).float().cuda()
    # logits2_std4 = torch.from_numpy(np.random.normal(1, 0.025, size=logits2.size(1))).float().cuda()
    # logits2a = logits2 * logits2_std1
    # logits2b = logits2 * logits2_std2
    # logits2c = logits2 * logits2_std3
    # logits2d = logits2 * logits2_std4
    # logits1a /= args.moco_t
    # logits1b /= args.moco_t
    # logits1c /= args.moco_t
    # logits1d /= args.moco_t
    # logits2a /= args.moco_t
    # logits2b /= args.moco_t
    # logits2c /= args.moco_t
    # logits2d /= args.moco_t

    # print("*************q_pred", q_pred.shape)  # torch.Size([batch, 表征维度])
    # print("*************logits1", logits1.shape)  # torch.Size([batch, 原型数量])

    # 对字典加变分：不在q_pred本身采样，而是在q_pred周围采样
    # 对embedding q
    q_pred_std1 = torch.from_numpy(np.random.normal(1, 0.025, size=q_pred.size(1))).float().cuda()
    q_pred_std2 = torch.from_numpy(np.random.normal(1, 0.025, size=q_pred.size(1))).float().cuda()
    # q_pred_std3 = torch.from_numpy(np.random.normal(1, 0.025, size=q_pred.size(1))).float().cuda()  # 生成四个与embedding同维度的
    # q_pred_std4 = torch.from_numpy(np.random.normal(1, 0.025, size=q_pred.size(1))).float().cuda()  # 均值为1方差为0.025的变分
    # print("q_pred.shape", q_pred.shape, q_pred.size(0), q_pred.size(1))  # torch.Size([32, 128]), 32, 128
    # print("q_pred_std.shape", q_pred_std.shape)  # torch.Size([128])
    q_pred1 = q_pred * q_pred_std1
    q_pred2 = q_pred * q_pred_std2
    # q_pred3 = q_pred * q_pred_std3  # 然后让原embedding与变分相乘
    # q_pred4 = q_pred * q_pred_std4  # 相当于在周围采样四个点
    # 对embedding k
    # k_pred_std1 = torch.from_numpy(np.random.normal(1, 0.025, size=k_pred.size(1))).float().cuda()
    # k_pred_std2 = torch.from_numpy(np.random.normal(1, 0.025, size=k_pred.size(1))).float().cuda()
    # k_pred_std3 = torch.from_numpy(np.random.normal(1, 0.025, size=k_pred.size(1))).float().cuda()  
    # k_pred_std4 = torch.from_numpy(np.random.normal(1, 0.025, size=k_pred.size(1))).float().cuda()  
    # k_pred1 = k_pred * k_pred_std1
    # k_pred2 = k_pred * k_pred_std2
    # k_pred3 = k_pred * k_pred_std3  
    # k_pred4 = k_pred * k_pred_std4  
    # 用变分过的embedding经过memory bank字典查找出来的结果
    _, _, logits11 = Memory_Bank(q_pred1)
    _, _, logits12 = Memory_Bank(q_pred2)
    # _, _, logits13 = Memory_Bank(q_pred3)
    # _, _, logits14 = Memory_Bank(q_pred4)
    # _, _, logits21 = Memory_Bank(k_pred1)
    # _, _, logits22 = Memory_Bank(k_pred2)
    # _, _, logits23 = Memory_Bank(k_pred3)
    # _, _, logits24 = Memory_Bank(k_pred4)
    logits11 /= args.moco_t
    logits12 /= args.moco_t
    # logits13 /= args.moco_t
    # logits14 /= args.moco_t
    # logits21 /= args.moco_t
    # logits22 /= args.moco_t
    # logits23 /= args.moco_t
    # logits24 /= args.moco_t




    # logits: Nx(1+K)
    with torch.no_grad():
        logits_keep1 = logits1.clone()
        logits_keep2 = logits2.clone()
    
    logits1 /= args.moco_t #固定值0.08
    logits2 /= args.moco_t 
    # print("******")
    # print(logits1[0])
    # print(torch.nn.functional.softmax(logits1,dim=1)[0])
    # print(torch.nn.functional.normalize(logits1, p=args.norm)[0])


    # 找到正样本的索引和标签
    with torch.no_grad():
        #swap relationship, im_k supervise im_q
        # 消融实验，去除双分支k->k_pred
        d_norm21, d21, check_logits1 = Memory_Bank(k)
        # 浅拷贝问题
        logits_fix1 = copy.deepcopy(check_logits1)
        check_logits1 = check_logits1.detach()

        # 在分层字典索引约束下查找正样本
        # step_size = 4
        # num_ranges = len(labels1_upper) 
        # ranges = [(step_size * labels1_upper[i].item(), 4 + step_size * labels1_upper[i].item()) for i in range(num_ranges)]
        # filter_index1 = []
        # for i in range(num_ranges):
        #     start = ranges[i][0]
        #     end = ranges[i][1]
        #     sub_range_argmax = torch.argmax(check_logits1[i][start:end], dim=0)
        #     filter_index1.append(sub_range_argmax)
        # filter_index1 = torch.stack(filter_index1)
        # tensor_elements = [item[0] for item in ranges]
        # target_tensor = torch.tensor(tensor_elements).cuda()
        # labels1_hir = filter_index1 + target_tensor

        # 原写法
        filter_index1 = torch.argmax(check_logits1, dim=1)
        labels1 = copy.deepcopy(filter_index1)
        filter_findex1 = torch.argmin(check_logits1, dim=1)
        flabels1 = copy.deepcopy(filter_findex1)
        
        for i in range(check_logits1.size(0)):
            check_logits1[i][filter_index1[i]] = 0
        filter_index12 = torch.argmax(check_logits1, dim=1)
        labels12 = copy.deepcopy(filter_index12)

        for i in range(check_logits1.size(0)):
            # print(check_logits1[i][filter_index12[i]])
            check_logits1[i][filter_index12[i]] = 0
        filter_index13 = torch.argmax(check_logits1, dim=1)
        labels13 = copy.deepcopy(filter_index13)

        for i in range(check_logits1.size(0)):
            # print(check_logits1[i][filter_index13[i]])
            check_logits1[i][filter_index13[i]] = 0
        filter_index14 = torch.argmax(check_logits1, dim=1)
        labels14 = copy.deepcopy(filter_index14)

        for i in range(check_logits1.size(0)):
            # print(check_logits1[i][filter_index12[i]])
            check_logits1[i][filter_index14[i]] = 0
        filter_index15 = torch.argmax(check_logits1, dim=1)
        labels15 = copy.deepcopy(filter_index15)

        for i in range(check_logits1.size(0)):
            # print(check_logits1[i][filter_index13[i]])
            check_logits1[i][filter_index15[i]] = 0
        filter_index16 = torch.argmax(check_logits1, dim=1)
        labels16 = copy.deepcopy(filter_index16)

        for i in range(check_logits1.size(0)):
            check_logits1[i][filter_index16[i]] = 0
        filter_index17 = torch.argmax(check_logits1, dim=1)
        labels17 = copy.deepcopy(filter_index17)

        for i in range(check_logits1.size(0)):
            check_logits1[i][filter_index17[i]] = 0
        filter_index18 = torch.argmax(check_logits1, dim=1)
        labels18 = copy.deepcopy(filter_index18)

        for i in range(check_logits1.size(0)):
            check_logits1[i][filter_index18[i]] = 0
        filter_index19 = torch.argmax(check_logits1, dim=1)
        labels19 = copy.deepcopy(filter_index19)

        for i in range(check_logits1.size(0)):
            check_logits1[i][filter_index19[i]] = 0
        filter_index110 = torch.argmax(check_logits1, dim=1)
        labels110 = copy.deepcopy(filter_index110)

        check_logits1 = logits_fix1
        

        '''
        labelMat = np.zeros((args.batch_size, args.cluster))
        for i in range(args.batch_size):
            labels_batch = np.argpartition(check_logits1[i], len(check_logits1[i])-4)[-4:]
            labelMat[i].flat[labels_batch[0]] = 0.25
            labelMat[i].flat[labels_batch[1]] = 0.25
            labelMat[i].flat[labels_batch[2]] = 0.25
            labelMat[i].flat[labels_batch[3]] = 0.25
        '''

        # print(check_logits1[0])
        # print(check_logits1[1])
        # print(labels_batch1)
        # print(labels_batch2)

        # print("***************labels1", labels1.shape, labels1)  # torch.Size([32])，即一个batch的表征进入memory bank后找到的最像自己的那个表征的索引
        # print("***************labels1[0]", labels1[0].shape, labels1[0])  # torch.Size([])标量
        # print("***************labels1[1]", labels1[1].shape, labels1[1])  # torch.Size([])标量
        # labels12 = filter_index12
        # labels13 = filter_index13
        # labels14 = filter_index14
        # labels15 = filter_index15
        # labels16 = filter_index16
        # print(logits1[0])
        # print(labels1[0])
        # print(labels12[0])
        # print(labels13[0])
        # print(labels14[0])
        



        # 此处能不能取前n个作为正样本呢，而不是argmax的那一个？
        # print("check_logits1[6]", check_logits1[6])

        d_norm22, d22, check_logits2 = Memory_Bank(q)
        # print("check_logits2.shape：", check_logits2.shape)  # torch.Size([32, 1024])
        check_logits2 = check_logits2.detach()
        # filter_index2 = torch.argmax(check_logits2, dim=1)
        # labels2 = filter_index2
        logits_fix2 = check_logits2
        filter_index2 = torch.argmax(check_logits2, dim=1)
        for i in range(check_logits2.size(0)):
            # print(check_logits2[i][filter_index2[i]])
            check_logits2[i][filter_index2[i]] = 0
        filter_index22 = torch.argmax(check_logits2, dim=1)
        for i in range(check_logits2.size(0)):
            # print(check_logits2[i][filter_index22[i]])
            check_logits2[i][filter_index22[i]] = 0
        filter_index23 = torch.argmax(check_logits2, dim=1)
        for i in range(check_logits2.size(0)):
            # print(check_logits2[i][filter_index23[i]])
            check_logits2[i][filter_index23[i]] = 0
        filter_index24 = torch.argmax(check_logits2, dim=1)
        check_logits2 = logits_fix2
        
        labels2 = filter_index2
        labels22 = filter_index22
        labels23 = filter_index23
        labels24 = filter_index24
        
    
    # ┌─────────────────────────────────────────┐
    # |             计算原caco损失 
    # └─────────────────────────────────────────┘
    # caco_loss_SwAV = (criterion(logits1, labels1)+criterion(logits2, labels2))#*args.moco_t

    # ┌─────────────────────────────────────────┐
    # |             计算单边caco损失 
    # └─────────────────────────────────────────┘
    caco_loss = criterion(logits1, labels1) #- criterion(logits1, flabels1) * 0.1
    # caco_loss2 = criterion(logits2, labels2)

    # ┌─────────────────────────────────────────┐
    # |             计算DCL损失 
    # └─────────────────────────────────────────┘
    criterion_dcl = DCLLoss()
    dclloss = criterion_dcl(q_pred, k) + criterion(logits1, labels1)
    # print("三个损失项： ", criterion_dcl(q_pred, k), criterion(logits1, labels1), dclloss)  # about 2,1 and 3


    # ┌─────────────────────────────────────────┐
    # |             计算KL regularizer损失 
    # └─────────────────────────────────────────┘ 
    rho = args.rho
    # 范数消融
    # norm2 = F.normalize(logits1, p=2)
    # norm3 = F.normalize(logits1, p=3)
    # interpolated_norm = (norm2.pow(2.5) + norm3.pow(2.5)) / 2  # Interpolation
    # normalized_tensor = F.normalize(interpolated_norm, p=2.5)
    # rho_hat = torch.mean(normalized_tensor, dim=0)

    # 对logits1求softmax而不是norm  OA: IP 93.11
    # rho_hat = torch.mean(torch.nn.functional.softmax(logits1,dim=1), dim=0)
    # 对logits1求softmax之后再做norm  OA:69.08
    # soft_logits1 = torch.nn.functional.softmax(logits1,dim=1)
    # rho_hat = torch.mean(torch.nn.functional.normalize(soft_logits1, p=args.norm), dim=0)
    # 原写法  OA:96.70
    rho_hat = torch.mean(torch.nn.functional.normalize(logits1, p=args.norm), dim=0)
    # H_logits1 = torch.log(logits1) * logits1
    # print(H_logits1.shape)
    # print(H_logits1)
    # prob_logits1 = 
    # print(rho_hat.shape)  # 256
    KL = rho_hat * torch.log(rho_hat / rho) + (1-rho_hat) * torch.log((1-rho_hat)/(1-rho))  # 逆向，采用
    # KL = rho * torch.log(rho / rho_hat) + (1-rho) * torch.log((1-rho)/(1-rho_hat))  # 正向
    KL_loss =  torch.sum(KL) 
    # KL_alpha = (1 / (1 + np.exp(KL_loss))) * (1 / (1 + np.exp(KL_loss))) * (1 / (1 + np.exp(KL_loss))) 
    
    # plogp  OA: IP 96.70 +-0.02
    H_rho = torch.sum(-rho_hat * torch.log(rho_hat))
    # KL_alpha = 1 / (1 + np.exp( H_rho))  
    KL_alpha = 0.00001
    # 除以logn OA: IP 95.49 +-0.00  DI 94.3060
    # H_rho = torch.sum(-rho_hat * torch.log(rho_hat)) / math.log(float(args.cluster))
    # KL_alpha = 1 / (1 + np.exp( H_rho))  


    # ┌─────────────────────────────────────────┐
    # |             计算分层caco损失 
    # └─────────────────────────────────────────┘
    # print("浅层字典选取正样本： ", labels1_upper)
    # print("深层字典选取正样本： ", labels1)
    caco_hier = criterion(logits1, labels1) + criterion(logits1_upper, labels1_upper) #+ simi_loss * 2 #+ KL_loss
    caco_hier_upper01 = criterion(logits1, labels1) + criterion(logits1_upper, labels1_upper) - criterion(logits1_upper, flabels1_upper) * 0.1
    caco_hier_deep005 = criterion(logits1, labels1) + criterion(logits1_upper, labels1_upper) - criterion(logits1, flabels1) * 0.05
    caco_hier_deep01 = criterion(logits1, labels1) + criterion(logits1_upper, labels1_upper) - criterion(logits1, flabels1) * 0.1
    caco_hier_deep015 = criterion(logits1, labels1) + criterion(logits1_upper, labels1_upper) - criterion(logits1, flabels1) * 0.15
    caco_hier_deep02 = criterion(logits1, labels1) + criterion(logits1_upper, labels1_upper) - criterion(logits1, flabels1) * 0.2
    caco_hier_deep025 = criterion(logits1, labels1) + criterion(logits1_upper, labels1_upper) - criterion(logits1, flabels1) * 0.25
    caco_hier_deep03 = criterion(logits1, labels1) + criterion(logits1_upper, labels1_upper) - criterion(logits1, flabels1) * 0.3
    caco_hier_002 = criterion(logits1, labels1) + criterion(logits1_upper, labels1_upper) - criterion(logits1, flabels1) * 0.02 - criterion(logits1_upper, flabels1_upper) * 0.02
    caco_hier_004 = criterion(logits1, labels1) + criterion(logits1_upper, labels1_upper) - criterion(logits1, flabels1) * 0.04 - criterion(logits1_upper, flabels1_upper) * 0.04
    caco_hier_005 = criterion(logits1, labels1) + criterion(logits1_upper, labels1_upper) - criterion(logits1, flabels1) * 0.05 - criterion(logits1_upper, flabels1_upper) * 0.05
    caco_hier_006 = criterion(logits1, labels1) + criterion(logits1_upper, labels1_upper) - criterion(logits1, flabels1) * 0.06 - criterion(logits1_upper, flabels1_upper) * 0.06
    caco_hier_008 = criterion(logits1, labels1) + criterion(logits1_upper, labels1_upper) - criterion(logits1, flabels1) * 0.08 - criterion(logits1_upper, flabels1_upper) * 0.08
    caco_hier_01 = criterion(logits1, labels1) + criterion(logits1_upper, labels1_upper) - criterion(logits1, flabels1) * 0.1 - criterion(logits1_upper, flabels1_upper) * 0.1
    caco_hier_015 = criterion(logits1, labels1) + criterion(logits1_upper, labels1_upper) - criterion(logits1, flabels1) * 0.15 - criterion(logits1_upper, flabels1_upper) * 0.15
    caco_hier_02 = criterion(logits1, labels1) + criterion(logits1_upper, labels1_upper) - criterion(logits1, flabels1) * 0.2 - criterion(logits1_upper, flabels1_upper) * 0.2
    caco_hier_025 = criterion(logits1, labels1) + criterion(logits1_upper, labels1_upper) - criterion(logits1, flabels1) * 0.25 - criterion(logits1_upper, flabels1_upper) * 0.25
    caco_hier_03 = criterion(logits1, labels1) + criterion(logits1_upper, labels1_upper) - criterion(logits1, flabels1) * 0.3 - criterion(logits1_upper, flabels1_upper) * 0.3
    caco_hier_04 = criterion(logits1, labels1) + criterion(logits1_upper, labels1_upper) - criterion(logits1, flabels1) * 0.4 - criterion(logits1_upper, flabels1_upper) * 0.4
    caco_hier_05 = criterion(logits1, labels1) + criterion(logits1_upper, labels1_upper) - criterion(logits1, flabels1) * 0.5 - criterion(logits1_upper, flabels1_upper) * 0.5
    caco_hier_06 = criterion(logits1, labels1) + criterion(logits1_upper, labels1_upper) - criterion(logits1, flabels1) * 0.6 - criterion(logits1_upper, flabels1_upper) * 0.6
    caco_hier_07 = criterion(logits1, labels1) + criterion(logits1_upper, labels1_upper) - criterion(logits1, flabels1) * 0.7 - criterion(logits1_upper, flabels1_upper) * 0.7
    caco_hier_08 = criterion(logits1, labels1) + criterion(logits1_upper, labels1_upper) - criterion(logits1, flabels1) * 0.8 - criterion(logits1_upper, flabels1_upper) * 0.8
    caco_hier_09 = criterion(logits1, labels1) + criterion(logits1_upper, labels1_upper) - criterion(logits1, flabels1) * 0.9 - criterion(logits1_upper, flabels1_upper) * 0.9 
    caco_hier_10 = criterion(logits1, labels1) + criterion(logits1_upper, labels1_upper) - criterion(logits1, flabels1) - criterion(logits1_upper, flabels1_upper)   
    # 深浅不同权
    caco_hier_00101 = criterion(logits1, labels1) + criterion(logits1_upper, labels1_upper) - criterion(logits1, flabels1) * 0.01 - criterion(logits1_upper, flabels1_upper) * 0.1
    caco_hier_00501 = criterion(logits1, labels1) + criterion(logits1_upper, labels1_upper) - criterion(logits1, flabels1) * 0.05 - criterion(logits1_upper, flabels1_upper) * 0.1
    caco_hier_000101 = criterion(logits1, labels1) + criterion(logits1_upper, labels1_upper) - criterion(logits1, flabels1) * 0.001 - criterion(logits1_upper, flabels1_upper) * 0.1
    caco_hier_000501 = criterion(logits1, labels1) + criterion(logits1_upper, labels1_upper) - criterion(logits1, flabels1) * 0.005 - criterion(logits1_upper, flabels1_upper) * 0.1
    # caco_hier = criterion(logits1, labels1) + criterion(logits1, labels1_hir)
    # caco_hier = criterion(logits1, labels1) + criterion(logits1, labels1_hir)
    # caco_hier = criterion(logits1_upper, labels1_upper)


    # ┌─────────────────────────────────────────┐
    # |        计算动态权重公式xhot KL损失 
    # └─────────────────────────────────────────┘
    Dy_x1 = logits1
    sim = torch.nn.functional.normalize(logits1)
    Dy_fourhot = np.zeros((args.batch_size, args.cluster))  # 不指定数据类型默认是float
    # print("查看标签向量形状", Dy_fourhot)
    for i in range(labels1.size(0)):
        if args.loss=="Dy2hot+KL" or args.loss=="2Dy2hot+KL" or args.loss=="3Dy2hot+KL" or args.loss=="4Dy2hot+KL" or args.loss=="Dy2hot+2KL" or args.loss=="Dy2hot+3KL" or args.loss=="Dy2hot+4KL" or args.loss=="Dy2hot+1.6KL" or args.loss=="Dy2hot+0.4KL" or args.loss=="Dy2hot":
            Dy_fourhot[i] = np.zeros((1, args.cluster))
            Dy_fourhot[i].flat[labels1[i]] = 1
            Dy_fourhot[i].flat[labels12[i]] = KL_alpha
        elif args.loss=="Dy3hot+KL" or args.loss=="2Dy3hot+KL" or args.loss=="3Dy3hot+KL" or args.loss=="4Dy3hot+KL" or args.loss=="Dy3hot+2KL" or args.loss=="Dy3hot+3KL" or args.loss=="Dy3hot+4KL" or args.loss=="Dy3hot+0.4KL" or args.loss=="Dy3hot+0.8KL" or args.loss=="Dy3hot+1.2KL" or args.loss=="Dy3hot+1.6KL" or args.loss=="Dy3hot+0.4KL" or args.loss=="Dy3hot":
            Dy_fourhot[i] = np.zeros((1, args.cluster))
            Dy_fourhot[i].flat[labels1[i]] = 1
            Dy_fourhot[i].flat[labels12[i]] = KL_alpha
            Dy_fourhot[i].flat[labels13[i]] = KL_alpha
        elif args.loss=="Dy4hot+KL" or args.loss=="2Dy4hot+KL" or args.loss=="3Dy4hot+KL" or args.loss=="4Dy4hot+KL" or args.loss=="5Dy4hot+KL" or args.loss=="6Dy4hot+KL" or args.loss=="7Dy4hot+KL" or args.loss=="8Dy4hot+KL" or args.loss=="Dy4hot+2KL" or args.loss=="Dy4hot+3KL" or args.loss=="Dy4hot+4KL" or args.loss=="Dy4hot+0.4KL" or args.loss=="Dy4hot+0.8KL" or args.loss=="Dy4hot+1.2KL" or args.loss=="Dy4hot+1.6KL" or args.loss=="Dy4hot+2.4KL" or args.loss=="Dy4hot2P+2KL" or args.loss=="Dy4hot3P+3KL" or args.loss=="Dy4hot":
            Dy_fourhot[i] = np.zeros((1, args.cluster))
            Dy_fourhot[i].flat[labels1[i]] = 1
            Dy_fourhot[i].flat[labels12[i]] = KL_alpha
            Dy_fourhot[i].flat[labels13[i]] = KL_alpha
            Dy_fourhot[i].flat[labels14[i]] = KL_alpha
        elif args.loss=="Dy5hot+KL" or args.loss=="2Dy5hot+KL" or args.loss=="3Dy5hot+KL" or args.loss=="4Dy5hot+KL" or args.loss=="Dy5hot+2KL" or args.loss=="Dy5hot+3KL" or args.loss=="Dy5hot+4KL" or args.loss=="Dy5hot+1.6KL" or args.loss=="Dy5hot+0.4KL" or args.loss=="Dy5hot":
            Dy_fourhot[i] = np.zeros((1, args.cluster))
            Dy_fourhot[i].flat[labels1[i]] = 1
            Dy_fourhot[i].flat[labels12[i]] = KL_alpha
            Dy_fourhot[i].flat[labels13[i]] = KL_alpha
            Dy_fourhot[i].flat[labels14[i]] = KL_alpha
            Dy_fourhot[i].flat[labels15[i]] = KL_alpha
        elif args.loss=="Dy6hot+KL" or args.loss=="2Dy6hot+KL" or args.loss=="3Dy6hot+KL" or args.loss=="4Dy6hot+KL" or args.loss=="Dy6hot+2KL" or args.loss=="Dy6hot+3KL" or args.loss=="Dy6hot+4KL" or args.loss=="Dy6hot+1.6KL" or args.loss=="Dy6hot+0.4KL" or args.loss=="Dy6hot":
            Dy_fourhot[i] = np.zeros((1, args.cluster))
            Dy_fourhot[i].flat[labels1[i]] = 1
            Dy_fourhot[i].flat[labels12[i]] = KL_alpha
            Dy_fourhot[i].flat[labels13[i]] = KL_alpha
            Dy_fourhot[i].flat[labels14[i]] = KL_alpha
            Dy_fourhot[i].flat[labels15[i]] = KL_alpha
            Dy_fourhot[i].flat[labels16[i]] = KL_alpha
        elif args.loss=="Dy7hot+1.6KL" or args.loss=="Dy7hot+0.4KL" or args.loss=="Dy7hot":
            Dy_fourhot[i] = np.zeros((1, args.cluster))
            Dy_fourhot[i].flat[labels1[i]] = 1
            Dy_fourhot[i].flat[labels12[i]] = KL_alpha
            Dy_fourhot[i].flat[labels13[i]] = KL_alpha
            Dy_fourhot[i].flat[labels14[i]] = KL_alpha
            Dy_fourhot[i].flat[labels15[i]] = KL_alpha
            Dy_fourhot[i].flat[labels16[i]] = KL_alpha
            Dy_fourhot[i].flat[labels17[i]] = KL_alpha
        elif args.loss=="Dy8hot+1.6KL" or args.loss=="Dy8hot+0.4KL" or args.loss=="Dy8hot":
            Dy_fourhot[i] = np.zeros((1, args.cluster))
            Dy_fourhot[i].flat[labels1[i]] = 1
            Dy_fourhot[i].flat[labels12[i]] = KL_alpha
            Dy_fourhot[i].flat[labels13[i]] = KL_alpha
            Dy_fourhot[i].flat[labels14[i]] = KL_alpha
            Dy_fourhot[i].flat[labels15[i]] = KL_alpha
            Dy_fourhot[i].flat[labels16[i]] = KL_alpha
            Dy_fourhot[i].flat[labels17[i]] = KL_alpha
            Dy_fourhot[i].flat[labels18[i]] = KL_alpha
        elif args.loss=="Dy9hot+1.6KL" or args.loss=="Dy9hot+0.4KL" or args.loss=="Dy9hot":
            Dy_fourhot[i] = np.zeros((1, args.cluster))
            Dy_fourhot[i].flat[labels1[i]] = 1
            Dy_fourhot[i].flat[labels12[i]] = KL_alpha
            Dy_fourhot[i].flat[labels13[i]] = KL_alpha
            Dy_fourhot[i].flat[labels14[i]] = KL_alpha
            Dy_fourhot[i].flat[labels15[i]] = KL_alpha
            Dy_fourhot[i].flat[labels16[i]] = KL_alpha
            Dy_fourhot[i].flat[labels17[i]] = KL_alpha
            Dy_fourhot[i].flat[labels18[i]] = KL_alpha
            Dy_fourhot[i].flat[labels19[i]] = KL_alpha
        elif args.loss=="Dy10hot+1.6KL" or args.loss=="Dy10hot":
            Dy_fourhot[i] = np.zeros((1, args.cluster))
            Dy_fourhot[i].flat[labels1[i]] = 1
            Dy_fourhot[i].flat[labels12[i]] = KL_alpha
            Dy_fourhot[i].flat[labels13[i]] = KL_alpha
            Dy_fourhot[i].flat[labels14[i]] = KL_alpha
            Dy_fourhot[i].flat[labels15[i]] = KL_alpha
            Dy_fourhot[i].flat[labels16[i]] = KL_alpha
            Dy_fourhot[i].flat[labels17[i]] = KL_alpha
            Dy_fourhot[i].flat[labels18[i]] = KL_alpha
            Dy_fourhot[i].flat[labels19[i]] = KL_alpha
            Dy_fourhot[i].flat[labels110[i]] = KL_alpha
        elif args.loss=="4hot+3KL":
            Dy_fourhot[i] = np.zeros((1, args.cluster))
            Dy_fourhot[i].flat[labels1[i]] = 1
            Dy_fourhot[i].flat[labels12[i]] = sim[i][labels12[i]] * sim[i][labels12[i]] * sim[i][labels12[i]]
            Dy_fourhot[i].flat[labels13[i]] = sim[i][labels13[i]] * sim[i][labels13[i]] * sim[i][labels13[i]]
            Dy_fourhot[i].flat[labels14[i]] = sim[i][labels14[i]] * sim[i][labels14[i]] * sim[i][labels14[i]]
        # elif args.loss=="caco_hier":
        #     Dy_fourhot[i] = np.zeros((1, args.cluster))
        #     Dy_fourhot[i].flat[labels1[i]] = 1
        #     Dy_fourhot[i].flat[labels1_upper[i]] = 0.005
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Dy_y1 = torch.tensor(Dy_fourhot).to(device)
    # print(Dy_y1)
    norm_y = torch.nn.functional.normalize(Dy_y1)  # 96.72 +-0.03
    # print(norm_y)
    x_hot_loss = x_hot_CrossEntropy()
    Dyxhot = x_hot_loss(Dy_x1,norm_y) 



    # ┌─────────────────────────────────────────┐
    # |                计算总损失 
    # └─────────────────────────────────────────┘
    if args.loss=="caco":
        loss = caco_loss
    elif args.loss=='caco_hier':
        loss = caco_hier
    elif args.loss=='caco_hier_002':
        loss = caco_hier_002
    elif args.loss=='caco_hier_004':
        loss = caco_hier_004
    elif args.loss=='caco_hier_005':
        loss = caco_hier_005
    elif args.loss=='caco_hier_006':
        loss = caco_hier_006
    elif args.loss=='caco_hier_008':
        loss = caco_hier_008
    elif args.loss=='caco_hier_01':
        loss = caco_hier_01
    elif args.loss=='caco_hier_015':
        loss = caco_hier_015
    elif args.loss=='caco_hier_02':
        loss = caco_hier_02
    elif args.loss=='caco_hier_025':
        loss = caco_hier_025
    elif args.loss=='caco_hier_03':
        loss = caco_hier_03
    elif args.loss=='caco_hier_04':
        loss = caco_hier_04
    elif args.loss=='caco_hier_05':
        loss = caco_hier_05
    elif args.loss=='caco_hier_06':
        loss = caco_hier_06
    elif args.loss=='caco_hier_07':
        loss = caco_hier_07
    elif args.loss=='caco_hier_08':
        loss = caco_hier_08
    elif args.loss=='caco_hier_09':
        loss = caco_hier_09
    elif args.loss=='caco_hier_10':
        loss = caco_hier_10
    elif args.loss=='caco_hier_upper01':
        loss = caco_hier_upper01
    elif args.loss=='caco_hier_deep005':
        loss = caco_hier_deep005
    elif args.loss=='caco_hier_deep01':
        loss = caco_hier_deep01
    elif args.loss=='caco_hier_deep015':
        loss = caco_hier_deep015
    elif args.loss=='caco_hier_deep02':
        loss = caco_hier_deep02
    elif args.loss=='caco_hier_deep025':
        loss = caco_hier_deep025
    elif args.loss=='caco_hier_deep03':
        loss = caco_hier_deep03
    elif args.loss=='caco_hier_00101':
        loss = caco_hier_00101
    elif args.loss=='caco_hier_00501':
        loss = caco_hier_00501
    elif args.loss=='caco_hier_000101':
        loss = caco_hier_000101
    elif args.loss=='caco_hier_000501':
        loss = caco_hier_000501
    elif args.loss=='hier_dcl':
        loss = dclloss
    # elif args.loss=="caco2":
    #     loss = caco_loss2
    # elif args.loss=="caco_SwAV":
    #     loss = caco_loss_SwAV
    elif args.loss=='caco+KL':
        loss = caco_loss + KL_loss
    elif args.loss=="Dy4hot+KL":
        loss = Dyxhot + KL_loss
    # elif args.loss=="caco+memax":
    #     loss = caco_loss + memaxloss
    # elif args.loss=="caco+2memax":
    #     loss = caco_loss + memaxloss * 2
    # elif args.loss=="caco+3memax":
    #     loss = caco_loss + memaxloss * 3
    # elif args.loss=="caco+4memax":
    #     loss = caco_loss + memaxloss * 4
    # elif args.loss=="2caco+memax":
    #     loss = caco_loss * 2 + memaxloss
    # elif args.loss=="3caco+memax":
    #     loss = caco_loss * 3 + memaxloss
    # elif args.loss=="4caco+memax":
    #     loss = caco_loss * 4 + memaxloss
    elif args.loss=="caco+0.4KL":
        loss = caco_loss + KL_loss * 0.4
    elif args.loss=="caco+0.8KL":
        loss = caco_loss + KL_loss * 0.8
    elif args.loss=="caco+1.2KL":
        loss = caco_loss + KL_loss * 1.2
    elif args.loss=="caco+1.6KL":
        loss = caco_loss + KL_loss * 1.6
    elif args.loss=="caco+2.4KL":
        loss = caco_loss + KL_loss * 2.4
    elif args.loss=="caco+2KL":
        loss = caco_loss + KL_loss * 2
    elif args.loss=="caco+3KL":
        loss = caco_loss + KL_loss * 3
    elif args.loss=="caco+4KL":
        loss = caco_loss + KL_loss * 4
    elif args.loss=="2caco+KL":
        loss = caco_loss * 2 + KL_loss
    elif args.loss=="3caco+KL":
        loss = caco_loss * 3 + KL_loss
    elif args.loss=="4caco+KL":
        loss = caco_loss * 4 + KL_loss
    elif args.loss=="5caco+KL":
        loss = caco_loss * 5 + KL_loss
    elif args.loss=="6caco+KL":
        loss = caco_loss * 6 + KL_loss
    elif args.loss=="7caco+KL":
        loss = caco_loss * 7 + KL_loss
    elif args.loss=="8caco+KL":
        loss = caco_loss * 8 + KL_loss
    # elif args.loss=="4hot+memax":
    #     loss = fourhot_loss + memaxloss
    # elif args.loss=="4po":
    #     loss = mpo_loss
    # elif args.loss=="caco+repetition":
    #     loss = caco_loss + repetition
    # elif args.loss=="Dy2hot" or args.loss=="Dy3hot" or args.loss=="Dy4hot" or args.loss=="Dy5hot" or args.loss=="Dy6hot":
    #     loss = Dyxhot
    # elif args.loss=="Dy4hot+repetition":
    #     loss = Dyxhot + repetition
    # elif args.loss=="4hot+repetition":
    #     loss = fourhot_loss + repetition
    # elif args.loss=="2hot+memax" or args.loss=="3hot+memax" or args.loss=="4hot+memax" or args.loss=="5hot+memax" or args.loss=="6hot+memax":
    #     loss = xhot_loss + memaxloss
    # elif args.loss=="Dy2hot+memax" or args.loss=="Dy3hot+memax" or args.loss=="Dy4hot+memax" or args.loss=="Dy5hot+memax" or args.loss=="Dy6hot+memax":
    #     loss = Dyxhot + memaxloss
    # elif args.loss=="Dy2hot+2memax" or args.loss=="Dy3hot+2memax" or args.loss=="Dy4hot+2memax" or args.loss=="Dy5hot+2memax" or args.loss=="Dy6hot+2memax":
    #     loss = Dyxhot + memaxloss * 2
    # elif args.loss=="Dy2hot+3memax" or args.loss=="Dy3hot+3memax" or args.loss=="Dy4hot+3memax" or args.loss=="Dy5hot+3memax" or args.loss=="Dy6hot+3memax":
    #     loss = Dyxhot + memaxloss * 3
    # elif args.loss=="Dy2hot+4memax" or args.loss=="Dy3hot+4memax" or args.loss=="Dy4hot+4memax" or args.loss=="Dy5hot+4memax" or args.loss=="Dy6hot+4memax":
    #     loss = Dyxhot + memaxloss * 4
    # elif args.loss=="2Dy2hot+memax" or args.loss=="2Dy3hot+memax" or args.loss=="2Dy4hot+memax" or args.loss=="2Dy5hot+memax" or args.loss=="2Dy6hot+memax":
    #     loss = Dyxhot * 2 + memaxloss
    # elif args.loss=="3Dy2hot+memax" or args.loss=="3Dy3hot+memax" or args.loss=="3Dy4hot+memax" or args.loss=="3Dy5hot+memax" or args.loss=="3Dy6hot+memax":
    #     loss = Dyxhot * 3 + memaxloss
    # elif args.loss=="4Dy2hot+memax" or args.loss=="4Dy3hot+memax" or args.loss=="4Dy4hot+memax" or args.loss=="4Dy5hot+memax" or args.loss=="4Dy6hot+memax":
    #     loss = Dyxhot * 4 + memaxloss
    elif args.loss=="Dy2hot+KL" or args.loss=="Dy3hot+KL" or args.loss=="Dy4hot+KL" or args.loss=="Dy5hot+KL" or args.loss=="Dy6hot+KL":
        loss = Dyxhot + KL_loss
    elif args.loss=="Dy2hot+2KL" or args.loss=="Dy3hot+2KL" or args.loss=="Dy4hot+2KL" or args.loss=="Dy5hot+2KL" or args.loss=="Dy6hot+2KL":
        loss = Dyxhot + KL_loss * 2
    elif args.loss=="Dy2hot+3KL" or args.loss=="Dy3hot+3KL" or args.loss=="Dy4hot+3KL" or args.loss=="Dy5hot+3KL" or args.loss=="Dy6hot+3KL":
        loss = Dyxhot + KL_loss * 3
    elif args.loss=="Dy2hot+4KL" or args.loss=="Dy3hot+4KL" or args.loss=="Dy4hot+4KL" or args.loss=="Dy5hot+4KL" or args.loss=="Dy6hot+4KL":
        loss = Dyxhot + KL_loss * 4
    elif args.loss=="2Dy2hot+KL" or args.loss=="2Dy3hot+KL" or args.loss=="2Dy4hot+KL" or args.loss=="2Dy5hot+KL" or args.loss=="2Dy6hot+KL":
        loss = Dyxhot * 2 + KL_loss
    elif args.loss=="3Dy2hot+KL" or args.loss=="3Dy3hot+KL" or args.loss=="3Dy4hot+KL" or args.loss=="3Dy5hot+KL" or args.loss=="3Dy6hot+KL":
        loss = Dyxhot * 3 + KL_loss
    elif args.loss=="4Dy2hot+KL" or args.loss=="4Dy3hot+KL" or args.loss=="4Dy4hot+KL" or args.loss=="4Dy5hot+KL" or args.loss=="4Dy6hot+KL":
        loss = Dyxhot * 4 + KL_loss
    elif args.loss=="5Dy2hot+KL" or args.loss=="5Dy3hot+KL" or args.loss=="5Dy4hot+KL" or args.loss=="5Dy5hot+KL" or args.loss=="5Dy6hot+KL":
        loss = Dyxhot * 5 + KL_loss
    elif args.loss=="6Dy2hot+KL" or args.loss=="6Dy3hot+KL" or args.loss=="6Dy4hot+KL" or args.loss=="6Dy5hot+KL" or args.loss=="6Dy6hot+KL":
        loss = Dyxhot * 6 + KL_loss
    elif args.loss=="7Dy2hot+KL" or args.loss=="7Dy3hot+KL" or args.loss=="7Dy4hot+KL" or args.loss=="7Dy5hot+KL" or args.loss=="7Dy6hot+KL":
        loss = Dyxhot * 7 + KL_loss
    elif args.loss=="8Dy2hot+KL" or args.loss=="8Dy3hot+KL" or args.loss=="8Dy4hot+KL" or args.loss=="8Dy5hot+KL" or args.loss=="8Dy6hot+KL":
        loss = Dyxhot * 8 + KL_loss
    elif args.loss=="4hot+3KL":
        loss = Dyxhot + KL_loss * 3
    elif args.loss=="Dy2hot+0.4KL" or args.loss=="Dy3hot+0.4KL" or args.loss=="Dy4hot+0.4KL"  or args.loss=="Dy5hot+0.4KL"  or args.loss=="Dy6hot+0.4KL"  or args.loss=="Dy7hot+0.4KL"  or args.loss=="Dy8hot+0.4KL"  or args.loss=="Dy9hot+0.4KL" :
        loss = Dyxhot + KL_loss * 0.4
    elif args.loss=="Dy4hot+0.8KL":
        loss = Dyxhot + KL_loss * 0.8
    elif args.loss=="Dy4hot+1.2KL":
        loss = Dyxhot + KL_loss * 1.2
    elif args.loss=="Dy2hot+1.6KL" or args.loss=="Dy3hot+1.6KL" or args.loss=="Dy4hot+1.6KL" or args.loss=="Dy5hot+1.6KL" or args.loss=="Dy6hot+1.6KL" or args.loss=="Dy7hot+1.6KL" or args.loss=="Dy8hot+1.6KL" or args.loss=="Dy9hot+1.6KL" or args.loss=="Dy10hot+1.6KL":
        loss = Dyxhot + KL_loss * 1.6
    elif args.loss=="Dy4hot+2.4KL":
        loss = Dyxhot + KL_loss * 2.4
    elif args.loss=="Dy4hot2P+2KL":
        loss = Dyxhot + KL_loss * 2
    elif args.loss=="Dy4hot3P+3KL":
        loss = Dyxhot + KL_loss * 3
    elif args.loss=="Dy2hot" or args.loss=="Dy3hot" or args.loss=="Dy4hot" or args.loss=="Dy5hot" or args.loss=="Dy6hot" or args.loss=="Dy7hot" or args.loss=="Dy8hot" or args.loss=="Dy9hot" or args.loss=="Dy10hot":
        loss = Dyxhot 
        


    # elif args.loss=="0.8Dy4hot+memax":
    #     loss = 0.8 * Dyxhot + 0.2 * memaxloss
    # elif args.loss=="mlg":
    #     loss = mlg_loss
    else:
        print("请设置loss超参!")
    # print("**********")
    # print("caco_loss", caco_loss)
    # print("KL_loss", KL_loss)
    

    # acc1/acc5 are (K+1)-way contrast classifier accuracy
    # measure accuracy and record loss
    acc1, acc5 = accuracy(logits1, labels1, topk=(1, 5))
    losses.update(loss.item(), images[0].size(0))
    top1.update(acc1.item(), images[0].size(0))
    top5.update(acc5.item(), images[0].size(0))
    acc1, acc5 = accuracy(logits2, labels2, topk=(1, 5))
    losses.update(loss.item(), images[0].size(0))
    top1.update(acc1.item(), images[0].size(0))
    top5.update(acc5.item(), images[0].size(0))
    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 添加更新小字典
    with torch.no_grad():
        logits1_upper = logits1_upper_keep/cur_adco_t#/args.mem_t
        logits2_upper = logits2_upper_keep/cur_adco_t#/args.mem_t
        
        p_qd1_upper = nn.functional.softmax(logits1_upper, dim=1)
        p_qd1_upper[torch.arange(logits1_upper.shape[0]), filter_index1_upper] = 1 - p_qd1_upper[torch.arange(logits1_upper.shape[0]), filter_index1_upper]
        g1_upper = torch.einsum('cn,nk->ck', [q_pred_upper.T, p_qd1_upper]) / logits1_upper.shape[0] - torch.mul(
            torch.mean(torch.mul(p_qd1_upper, logits1_upper_keep), dim=0), d_norm1_upper)
        
        p_qd2_upper = nn.functional.softmax(logits2_upper, dim=1)
        p_qd2_upper[torch.arange(logits1_upper.shape[0]), filter_index2_upper] = 1 - p_qd2_upper[torch.arange(logits2_upper.shape[0]), filter_index2_upper]
        g2_upper = torch.einsum('cn,nk->ck', [k_pred_upper.T, p_qd2_upper]) / logits2_upper.shape[0] - torch.mul(
            torch.mean(torch.mul(p_qd2_upper, logits2_upper_keep), dim=0), d_norm2_upper)
        g_upper = -torch.div(g1_upper, torch.norm(d1_upper, dim=0))  - torch.div(g2_upper,torch.norm(d2_upper, dim=0))#/ args.mem_t  # c*k
        g_upper /=cur_adco_t
        
        Memory_Bank_upper.v.data = args.mem_momentum * Memory_Bank_upper.v.data + g_upper #+ args.mem_wd * Memory_Bank.W.data
        Memory_Bank_upper.W.data = Memory_Bank_upper.W.data - memory_lr * Memory_Bank_upper.v.data
    
    
    # update memory bank
    with torch.no_grad():
        # 更新memory bank
        # logits: Nx(1+K)
        logits1 = logits_keep1/cur_adco_t#/args.mem_t
        # negative logits: NxK
        # logits: Nx(1+K)
        logits2 = logits_keep2/cur_adco_t#/args.mem_t
        
        p_qd1 = nn.functional.softmax(logits1, dim=1)
        p_qd1[torch.arange(logits1.shape[0]), filter_index1] = 1 - p_qd1[torch.arange(logits1.shape[0]), filter_index1]
        # 增加top-4正样本梯度回传
        # p_qd1[torch.arange(logits1.shape[0]), filter_index12] = 1 - p_qd1[torch.arange(logits1.shape[0]), filter_index12]
        # p_qd1[torch.arange(logits1.shape[0]), filter_index13] = 1 - p_qd1[torch.arange(logits1.shape[0]), filter_index13]
        # p_qd1[torch.arange(logits1.shape[0]), filter_index14] = 1 - p_qd1[torch.arange(logits1.shape[0]), filter_index14]

        g1 = torch.einsum('cn,nk->ck', [q_pred.T, p_qd1]) / logits1.shape[0] - torch.mul(
            torch.mean(torch.mul(p_qd1, logits_keep1), dim=0), d_norm1)
        
        
        p_qd2 = nn.functional.softmax(logits2, dim=1)
        p_qd2[torch.arange(logits1.shape[0]), filter_index2] = 1 - p_qd2[torch.arange(logits2.shape[0]), filter_index2]
        # 增加top-4正样本梯度回传
        # p_qd2[torch.arange(logits1.shape[0]), filter_index22] = 1 - p_qd2[torch.arange(logits2.shape[0]), filter_index22]
        # p_qd2[torch.arange(logits1.shape[0]), filter_index23] = 1 - p_qd2[torch.arange(logits2.shape[0]), filter_index23]
        # p_qd2[torch.arange(logits1.shape[0]), filter_index24] = 1 - p_qd2[torch.arange(logits2.shape[0]), filter_index24]


        g2 = torch.einsum('cn,nk->ck', [k_pred.T, p_qd2]) / logits2.shape[0] - torch.mul(
            torch.mean(torch.mul(p_qd2, logits_keep2), dim=0), d_norm2)
        g = -torch.div(g1, torch.norm(d1, dim=0))  - torch.div(g2,torch.norm(d2, dim=0))#/ args.mem_t  # c*k
        g /=cur_adco_t
        
        # g = all_reduce(g) / torch.distributed.get_world_size()
        Memory_Bank.v.data = args.mem_momentum * Memory_Bank.v.data + g #+ args.mem_wd * Memory_Bank.W.data
        Memory_Bank.W.data = Memory_Bank.W.data - memory_lr * Memory_Bank.v.data
    
    with torch.no_grad():
        logits = torch.softmax(logits1, dim=1)
        posi_prob = logits[torch.arange(logits.shape[0]), filter_index1]
        # 增加top-4正样本梯度回传
        # posi_prob2 = logits[torch.arange(logits.shape[0]), filter_index12]
        # posi_prob3 = logits[torch.arange(logits.shape[0]), filter_index13]
        # posi_prob4 = logits[torch.arange(logits.shape[0]), filter_index14]
        # print("***************", posi_prob)
        # print("###############", posi_prob2)
        # print("$$$$$$$$$$$$$$$", posi_prob3)
        # print("&&&&&&&&&&&&&&&", posi_prob4)
        posi_prob = torch.mean(posi_prob)
        # posi_prob = torch.mean(0.25* (posi_prob + posi_prob2 + posi_prob3 + posi_prob4))
        mem_losses.update(posi_prob.item(), logits.size(0))
    
    
    # k.shape:torch.Size([64, 64])  # 第一个64是batch，第二个是设定的经过model出来的维度
    # return logits2, logits1, q_pred, HB.item(), alpha.item(), labels1.tolist()
    # return logits2, logits1, q_pred, KL_loss.item(), KL_alpha.item(), labels1.tolist()
    return logits2, logits1, q_pred, KL_loss.item(), KL_alpha, labels1.tolist()

# 未使用
def update_symkey_network(model, images, args, Memory_Bank, 
                   losses, top1, top5, optimizer, criterion, mem_losses,
                   moco_momentum,memory_lr,cur_adco_t):
    model.zero_grad()
    q_pred, k_pred, q, k = model(im_q=images[0], im_k=images[1],run_type=0,moco_momentum=moco_momentum)
    
    d_norm1, d1, logits1 = Memory_Bank(q_pred)
    d_norm2, d2, logits2 = Memory_Bank(k_pred)
    # logits: Nx(1+K)
    with torch.no_grad():
        logits_keep1 = logits1.clone()
        logits_keep2 = logits2.clone()
    logits1 /= args.moco_t #cur_adco_t#args.moco_t
    logits2 /= args.moco_t #cur_adco_t#args.moco_t
    #find the positive index and label
    with torch.no_grad():
        #swap relationship, im_k supervise im_q
        d_norm21, d21, check_logits1 = Memory_Bank(k)
        check_logits1 = check_logits1.detach()
        filter_index1 = torch.argmax(check_logits1, dim=1)
        labels1 = filter_index1

        d_norm22, d22, check_logits2 = Memory_Bank(q)
        check_logits2 = check_logits2.detach()
        filter_index2 = torch.argmax(check_logits2, dim=1)
        labels2 = filter_index2

    # 损失函数在此，此处可做修改
    loss = (criterion(logits1, labels1)+criterion(logits2, labels2))#*args.moco_t
    # loss = criterion(logits1, logits2)


    # acc1/acc5 are (K+1)-way contrast classifier accuracy
    # measure accuracy and record loss
    acc1, acc5 = accuracy(logits1, labels1, topk=(1, 5))
    losses.update(loss.item(), images[0].size(0))
    top1.update(acc1.item(), images[0].size(0))
    top5.update(acc5.item(), images[0].size(0))
    acc1, acc5 = accuracy(logits2, labels2, topk=(1, 5))
    losses.update(loss.item(), images[0].size(0))
    top1.update(acc1.item(), images[0].size(0))
    top5.update(acc5.item(), images[0].size(0))

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # update memory bank
    with torch.no_grad():
        # update memory bank

        # logits: Nx(1+K)
        logits1 = check_logits1/cur_adco_t#/args.mem_t
        # negative logits: NxK
        # logits: Nx(1+K)
        logits2 = check_logits2/cur_adco_t#/args.mem_t
        
        p_qd1 = nn.functional.softmax(logits1, dim=1)
        p_qd1[torch.arange(logits1.shape[0]), filter_index1] = 1 - p_qd1[torch.arange(logits1.shape[0]), filter_index1]

        g1 = torch.einsum('cn,nk->ck', [q.T, p_qd1]) / logits1.shape[0] - torch.mul(
            torch.mean(torch.mul(p_qd1, check_logits1), dim=0), d_norm21)
        
        
        p_qd2 = nn.functional.softmax(logits2, dim=1)
        p_qd2[torch.arange(logits1.shape[0]), filter_index2] = 1 - p_qd2[torch.arange(logits2.shape[0]), filter_index2]



        g2 = torch.einsum('cn,nk->ck', [k.T, p_qd2]) / logits2.shape[0] - torch.mul(
            torch.mean(torch.mul(p_qd2, check_logits2), dim=0), d_norm22)
        g = -torch.div(g1, torch.norm(d21, dim=0))  - torch.div(g2,torch.norm(d22, dim=0))#/ args.mem_t  # c*k
        g /=cur_adco_t
        
        # g = all_reduce(g) / torch.distributed.get_world_size()
        Memory_Bank.v.data = args.mem_momentum * Memory_Bank.v.data + g #+ args.mem_wd * Memory_Bank.W.data
        Memory_Bank.W.data = Memory_Bank.W.data - memory_lr * Memory_Bank.v.data
    with torch.no_grad():
        logits = torch.softmax(logits1, dim=1)
        posi_prob = logits[torch.arange(logits.shape[0]), filter_index1]
        posi_prob = torch.mean(posi_prob)
        mem_losses.update(posi_prob.item(), logits.size(0))
    return logits2,logits1


# 仅当使用multicrop增强时才启用
def update_multicrop_network(model, images, args, Memory_Bank,
                       losses, top1, top5, optimizer, criterion, mem_losses,
                       moco_momentum, memory_lr, cur_adco_t):
    model.zero_grad()
    q_list, k_list = model(im_q=images[1:], im_k=images[0], run_type=1, moco_momentum=moco_momentum)
    pred_list = []
    for q_pred in q_list:
        d_norm1, d1, logits1 = Memory_Bank(q_pred)
        pred_list.append(logits1)
    # logits: Nx(1+K)
    logits_keep_list = []
    with torch.no_grad():
        for logits_tmp in pred_list:
            logits_keep_list.append(logits_tmp.clone())
    for k in range(len(pred_list)):
        pred_list[k]/=args.moco_t

    # find the positive index and label
    labels_list = []
    with torch.no_grad():
        # swap relationship, im_k supervise im_q
        for key in k_list:

            d_norm2, d2, check_logits1 = Memory_Bank(key)
            check_logits1 = check_logits1.detach()
            filter_index1 = torch.argmax(check_logits1, dim=1)
            labels1 = filter_index1
            labels_list.append(labels1)

    loss_big = 0
    loss_mini = 0
    count_big = 0
    count_mini = 0
    for i in range(len(pred_list)):

        for j in range(len(labels_list)):
            if i==j:
                continue
            if i<2:
                loss_big += criterion(pred_list[i],labels_list[j])
                count_big += 1
            else:
                loss_mini += criterion(pred_list[i],labels_list[j])
                count_mini +=1
            if i==0:
                # acc1/acc5 are (K+1)-way contrast classifier accuracy
                # measure accuracy and record loss
                acc1, acc5 = accuracy(pred_list[i], labels_list[j], topk=(1, 5))
                losses.update(loss_big.item(), images[0].size(0))
                top1.update(acc1.item(), images[0].size(0))
                top5.update(acc5.item(), images[0].size(0))
    if count_big!=0:
        loss_big = loss_big/count_big
    if count_mini!=0:
        loss_mini = loss_mini/count_mini
    loss = loss_big+loss_mini  # *args.moco_t


    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # update memory bank
    with torch.no_grad():
        # update memory bank

        g_big_sum = 0
        g_mini_sum = 0
        count_big = 0
        count_mini = 0
        for i in range(len(pred_list)):

            for j in range(len(labels_list)):
                if i == j:
                    continue
                logits1 = logits_keep_list[i]/cur_adco_t
                p_qd1 = nn.functional.softmax(logits1, dim=1)
                filter_index1 = labels_list[j]
                p_qd1[torch.arange(logits1.shape[0]), filter_index1] = 1 - p_qd1[torch.arange(logits1.shape[0]), filter_index1]
                q_pred = q_list[i]
                logits_keep1 = logits_keep_list[i]
                g1 = torch.einsum('cn,nk->ck', [q_pred.T, p_qd1]) / logits1.shape[0] - torch.mul(
                    torch.mean(torch.mul(p_qd1, logits_keep1), dim=0), d_norm1)


                g = -torch.div(g1, torch.norm(d1, dim=0))
                g /= cur_adco_t

                # g = all_reduce(g) / torch.distributed.get_world_size()
                if i<2:
                    g_big_sum +=g
                    count_big +=1
                else:
                    g_mini_sum += g
                    count_mini += 1
        if count_big != 0:
            g_big_sum = g_big_sum / count_big
        if count_mini != 0:
            g_mini_sum = g_mini_sum/ count_mini
        g_sum = g_big_sum +g_mini_sum
        Memory_Bank.v.data = args.mem_momentum * Memory_Bank.v.data + g_sum  # + args.mem_wd * Memory_Bank.W.data
        Memory_Bank.W.data = Memory_Bank.W.data - memory_lr * Memory_Bank.v.data
    with torch.no_grad():
        logits1 = pred_list[0]
        filter_index1 = labels_list[0]
        logits = torch.softmax(logits1, dim=1)
        posi_prob = logits[torch.arange(logits.shape[0]), filter_index1]
        posi_prob = torch.mean(posi_prob)
        mem_losses.update(posi_prob.item(), logits.size(0))
    return pred_list

@torch.no_grad()
def all_reduce(tensor):
    """
    Performs all_reduce(mean) operation on the provided tensors.
    *** Warning ***: torch.distributed.all_reduce has no gradient.
    """
    torch.distributed.all_reduce(tensor, async_op=False)

    return tensor


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


class AllReduce(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous() / dist.get_world_size()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads
    
# query:          [batch, 表征维度][32, 128]
# supports:       [表征维度,原型数量][128,256]
# support_labels: [原型数量，原型数量][256, 256]
# return:   [batch, 原型数量][32, 256]
# q_pred, Memory_Bank.W, proto_labels
def snn(query, supports, support_labels, temp=0.1):
    """ Soft Nearest Neighbours similarity classifier """
    # print("*******************************", query.shape, supports.shape, support_labels.shape)
    softmax = torch.nn.Softmax(dim=1)
    query = torch.nn.functional.normalize(query)
    supports = torch.nn.functional.normalize(supports)
    return softmax(query @ supports / temp) @ support_labels
    
def one_hot(targets, num_classes, smoothing=0):
    device = torch.device('cuda:0')
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    targets = targets.long().view(-1, 1).to(device)
    return torch.full((len(targets), num_classes), off_value, device=device).scatter_(1, targets, on_value)

def sharpen(p, T):
    sharp_p = p**(1./T)
    sharp_p /= torch.sum(sharp_p, dim=1, keepdim=True)
    return sharp_p

@torch.no_grad()
def distributed_sinkhorn(Q, num_itr=3, use_dist=True):
    _got_dist = use_dist and torch.distributed.is_available() \
        and torch.distributed.is_initialized() \
        and (torch.distributed.get_world_size() > 1)

    if _got_dist:
        world_size = torch.distributed.get_world_size()
    else:
        world_size = 1

    Q = Q.T
    B = Q.shape[1] * world_size  # number of samples to assign
    K = Q.shape[0]  # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    if _got_dist:
        torch.distributed.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(num_itr):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        if _got_dist:
            torch.distributed.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B  # the colomns must sum to 1 so that Q is an assignment
    return Q.T

class x_hot_CrossEntropy(torch.nn.Module):
    def __init__(self):
        super(x_hot_CrossEntropy,self).__init__()
    
    def forward(self, x ,y):
        P_i = torch.nn.functional.softmax(x, dim=1)
        # y = torch.nn.functional.softmax(y, dim=1)
        # print(y[0])
        loss = y*torch.log(P_i + 0.0000001)
        loss = -torch.mean(torch.sum(loss,dim=1),dim = 0)
        return loss
    
    # ┌─────────────────────────────────────────┐
    # |                损失留档 
    # └─────────────────────────────────────────┘
    # 使用Masked Siamese Networks的损失形式
    # q_pred: [batch, 表征维度]
    # Memory_Bank.W: [表征维度, 原型数量]
    # proto_labels: [原型数量，原型数量]
    # probs = q_pred @ Memory_Bank.W  # [batch,表征维度] @ [表征维度, 原型数量] = [batch, 原型数量]
    # targets = k @ Memory_Bank.W  # [batch,表征维度] @ [表征维度, 原型数量] = [batch, 原型数量]
    # ┌─────────────────────────────────────────┐
    # |       计算me-max regularizer损失 
    # └─────────────────────────────────────────┘
    # proto_labels = one_hot(torch.tensor([i for i in range(args.cluster)]), args.cluster)
    # # for i in range(proto_labels.size(0)):
    # #     print(proto_labels[i])
    # probs = snn(q_pred, Memory_Bank.W, proto_labels)
    # with torch.no_grad():
    #     targets = snn(k, Memory_Bank.W, proto_labels)
    #     targets = targets.detach()
    #     targets = sharpen(targets, T=0.25) 
    #     targets = distributed_sinkhorn(targets)
    #     targets = torch.cat([targets for _ in range(1)], dim=0)
    #     for i in range(targets[0].size(0)):
    #         print("target[0][" +str(i)+ "]", targets[0][i])
    # bloss = torch.mean(torch.sum(torch.log(probs**(-targets)), dim=1))  # cross-entropy损失H(targets, queries)
    # avg_probs = AllReduce.apply(torch.mean(probs, dim=0))
    # rloss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))  
    # memax_loss = bloss + rloss


    # ┌─────────────────────────────────────────┐
    # | 计算取多个正样本的me-max regularizer损失 
    # └─────────────────────────────────────────┘
    # proto_labels = one_hot(torch.tensor([i for i in range(args.cluster)]), args.cluster)
    # probs1 = snn(q_pred1, Memory_Bank.W, proto_labels)
    # probs2 = snn(q_pred2, Memory_Bank.W, proto_labels)
    # probs3 = snn(q_pred3, Memory_Bank.W, proto_labels)
    # probs4 = snn(q_pred4, Memory_Bank.W, proto_labels)
    # with torch.no_grad():
    #     targets = sharpen(snn(k, Memory_Bank.W, proto_labels), T=0.25) 
    #     targets = distributed_sinkhorn(targets)
    #     targets = torch.cat([targets for _ in range(1)], dim=0)
    # bloss1 = torch.mean(torch.sum(torch.log(probs1**(-targets)), dim=1))
    # bloss2 = torch.mean(torch.sum(torch.log(probs2**(-targets)), dim=1))
    # bloss3 = torch.mean(torch.sum(torch.log(probs3**(-targets)), dim=1))
    # bloss4 = torch.mean(torch.sum(torch.log(probs4**(-targets)), dim=1))
    # avg_probs1 = AllReduce.apply(torch.mean(probs1, dim=0))
    # avg_probs2 = AllReduce.apply(torch.mean(probs2, dim=0))
    # avg_probs3 = AllReduce.apply(torch.mean(probs3, dim=0))
    # avg_probs4 = AllReduce.apply(torch.mean(probs4, dim=0))
    # rloss1 = - torch.sum(torch.log(avg_probs1**(-avg_probs1))) + math.log(float(len(avg_probs1)))
    # rloss2 = - torch.sum(torch.log(avg_probs2**(-avg_probs2))) + math.log(float(len(avg_probs2)))
    # rloss3 = - torch.sum(torch.log(avg_probs3**(-avg_probs3))) + math.log(float(len(avg_probs3)))
    # rloss4 = - torch.sum(torch.log(avg_probs4**(-avg_probs4))) + math.log(float(len(avg_probs4)))
    # mmemax_loss = (bloss1 + rloss1) + (bloss2 + rloss2) + (bloss3 + rloss3) + (bloss4 + rloss4)


    # ┌─────────────────────────────────────────┐
    # |            计算欧氏距离损失 
    # └─────────────────────────────────────────┘
    # euc_loss = criterion(q_pred, k)


    # ┌─────────────────────────────────────────┐
    # |         计算取多个正样本的损失 
    # └─────────────────────────────────────────┘
    # 算四次，但是作为负样本的对象取0.5权重
    # 这种取法有一个要注意的点，就是因为labels是one-hot的，算第二三四次损失时候取的正样本作为了第一次正样本的负样本，可以考虑改成4-hot做对比
    # mpo_loss = (criterion(logits1, labels1)+criterion(logits1, labels12)+criterion(logits1, labels13)+criterion(logits1, labels14))+  \
    #         (criterion(logits2, labels2)+criterion(logits2, labels22)+criterion(logits2, labels23)+criterion(logits2, labels24))
    

    # ┌─────────────────────────────────────────┐
    # |     计算取多个字典embedding的损失 
    # └─────────────────────────────────────────┘
    # mcl_loss = (criterion(logits11, labels1)+criterion(logits12, labels1)+criterion(logits13, labels1)+criterion(logits14, labels1)+
    #         criterion(logits21, labels2)+criterion(logits22, labels2)+criterion(logits23, labels2)+criterion(logits24, labels2))


    # ┌─────────────────────────────────────────┐
    # |    计算取多个字典logit加vae变分的损失 
    # └─────────────────────────────────────────┘
    # LogitVae_loss = (criterion(logits1a, labels1)+criterion(logits1b, labels1)+criterion(logits1c, labels1)+criterion(logits1d, labels1)+
    #         criterion(logits2a, labels2)+criterion(logits2b, labels2)+criterion(logits2c, labels2)+criterion(logits2d, labels2))


    # ┌─────────────────────────────────────────┐
    # |     计算取多个正样本的4-hot损失v1 
    # └─────────────────────────────────────────┘
    # test_loss = criterion(logits1, labels1)
    # x1 = logits1
    # fourhot = np.zeros((32, 256))
    # for i in range(labels1.size(0)):
    #     # print(labels1[i])
    #     fourhot[i] = np.zeros((1, 256))
    #     fourhot[i].flat[labels1[i]] = 1
    #     fourhot[i].flat[labels12[i]] = 1
    #     fourhot[i].flat[labels13[i]] = 1
    #     fourhot[i].flat[labels14[i]] = 1
    # # for j in range(32):
    # #     print(fourhot[j])
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # y1 = torch.tensor(fourhot).to(device)
    # x_hot_loss = x_hot_CrossEntropy()
    # fourhot_loss_v1 = x_hot_loss(x1,y1)  # 加SwAV

    
    # ┌─────────────────────────────────────────┐
    # |     计算取多个正样本的4-hot损失v2 
    # └─────────────────────────────────────────┘
    # test_loss = criterion(logits1, labels1)
    # x1 = logits1
    # x2 = logits2
    # fourhot1 = np.zeros((32, 256))
    # fourhot12 = np.zeros((32, 256))
    # fourhot13 = np.zeros((32, 256))
    # fourhot14 = np.zeros((32, 256))
    # for i in range(labels1.size(0)):
    #     # print(labels1[i])
    #     fourhot1[i] = np.zeros((1, 256))
    #     fourhot1[i].flat[labels1[i]] = 1
    #     fourhot1[i].flat[labels12[i]] = 0.5
    #     fourhot1[i].flat[labels13[i]] = 0.5
    #     fourhot1[i].flat[labels14[i]] = 0.5
    #     fourhot12[i] = np.zeros((1, 256))
    #     fourhot12[i].flat[labels1[i]] = 0.5
    #     fourhot12[i].flat[labels12[i]] = 1
    #     fourhot12[i].flat[labels13[i]] = 0.5
    #     fourhot12[i].flat[labels14[i]] = 0.5
    #     fourhot13[i] = np.zeros((1, 256))
    #     fourhot13[i].flat[labels1[i]] = 0.5
    #     fourhot13[i].flat[labels12[i]] = 0.5
    #     fourhot13[i].flat[labels13[i]] = 1
    #     fourhot13[i].flat[labels14[i]] = 0.5
    #     fourhot14[i] = np.zeros((1, 256))
    #     fourhot14[i].flat[labels1[i]] = 0.5
    #     fourhot14[i].flat[labels12[i]] = 0.5
    #     fourhot14[i].flat[labels13[i]] = 0.5
    #     fourhot14[i].flat[labels14[i]] = 1
    # fourhot2 = np.zeros((32, 256))
    # fourhot22 = np.zeros((32, 256))
    # fourhot23 = np.zeros((32, 256))
    # fourhot24 = np.zeros((32, 256))
    # for i in range(labels2.size(0)):
    #     # print(labels1[i])
    #     fourhot2[i] = np.zeros((1, 256))
    #     fourhot2[i].flat[labels2[i]] = 1
    #     fourhot2[i].flat[labels22[i]] = 0.5
    #     fourhot2[i].flat[labels23[i]] = 0.5
    #     fourhot2[i].flat[labels24[i]] = 0.5
    #     fourhot22[i] = np.zeros((1, 256))
    #     fourhot22[i].flat[labels2[i]] = 0.5
    #     fourhot22[i].flat[labels22[i]] = 1
    #     fourhot22[i].flat[labels23[i]] = 0.5
    #     fourhot22[i].flat[labels24[i]] = 0.5
    #     fourhot23[i] = np.zeros((1, 256))
    #     fourhot23[i].flat[labels2[i]] = 0.5
    #     fourhot23[i].flat[labels22[i]] = 0.5
    #     fourhot23[i].flat[labels23[i]] = 1
    #     fourhot23[i].flat[labels24[i]] = 0.5
    #     fourhot24[i] = np.zeros((1, 256))
    #     fourhot24[i].flat[labels2[i]] = 0.5
    #     fourhot24[i].flat[labels22[i]] = 0.5
    #     fourhot24[i].flat[labels23[i]] = 0.5
    #     fourhot24[i].flat[labels24[i]] = 1
    # # for j in range(32):
    # #     print(fourhot[j])
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # y1 = torch.tensor(fourhot1).to(device)
    # y12 = torch.tensor(fourhot12).to(device)
    # y13 = torch.tensor(fourhot13).to(device)
    # y14 = torch.tensor(fourhot14).to(device)
    # y2 = torch.tensor(fourhot2).to(device)
    # y22 = torch.tensor(fourhot22).to(device)
    # y23 = torch.tensor(fourhot23).to(device)
    # y24 = torch.tensor(fourhot24).to(device)
    # x_hot_loss = x_hot_CrossEntropy()
    # fourhot_loss_v2 = x_hot_loss(x1,y1) + x_hot_loss(x1,y12) + x_hot_loss(x1,y13) + x_hot_loss(x1,y14) 
    #                 #   x_hot_loss(x2,y2) + x_hot_loss(x2,y22) + x_hot_loss(x2,y23) + x_hot_loss(x2,y24)

    # ┌─────────────────────────────────────────┐
    # |     计算取多个正样本的4-hot损失v3 
    # └─────────────────────────────────────────┘
    # test_loss = criterion(logits1, labels1)
    # x1 = logits1
    # x2 = logits2
    # fourhot = np.zeros((32, 256))
    # for i in range(labels1.size(0)):
    #     # print(labels1[i])
    #     fourhot[i] = np.zeros((1, 256))
    #     fourhot[i].flat[labels1[i]] = 1
    #     fourhot[i].flat[labels12[i]] = 0.5
    #     fourhot[i].flat[labels13[i]] = 0.5
    #     fourhot[i].flat[labels14[i]] = 0.5
    # fourhot2 = np.zeros((32, 256))
    # for i in range(labels2.size(0)):
    #     # print(labels1[i])
    #     fourhot2[i] = np.zeros((1, 256))
    #     fourhot2[i].flat[labels2[i]] = 1
    #     fourhot2[i].flat[labels22[i]] = 0.5
    #     fourhot2[i].flat[labels23[i]] = 0.5
    #     fourhot2[i].flat[labels24[i]] = 0.5
    # # for j in range(32):
    # #     print(fourhot[j])
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # y1 = torch.tensor(fourhot).to(device)
    # y2 = torch.tensor(fourhot2).to(device)
    # x_hot_loss = x_hot_CrossEntropy()
    # fourhot_loss_v3 = x_hot_loss(x1,y1)  + x_hot_loss(x2,y2) # 加SwAV


    # ┌─────────────────────────────────────────┐
    # |     计算4-hot v3损失+me-max损失改 
    # └─────────────────────────────────────────┘
    # x1 = logits1
    # x2 = logits2
    # fourhot = np.zeros((32, 256))
    # for i in range(labels1.size(0)):
    #     # print(labels1[i])
    #     fourhot[i] = np.zeros((1, 256))
    #     fourhot[i].flat[labels1[i]] = 1
    #     fourhot[i].flat[labels12[i]] = 0.5
    #     fourhot[i].flat[labels13[i]] = 0.5
    #     fourhot[i].flat[labels14[i]] = 0.5
    # fourhot2 = np.zeros((32, 256))
    # for i in range(labels2.size(0)):
    #     # print(labels1[i])
    #     fourhot2[i] = np.zeros((1, 256))
    #     fourhot2[i].flat[labels2[i]] = 1
    #     fourhot2[i].flat[labels22[i]] = 0.5
    #     fourhot2[i].flat[labels23[i]] = 0.5
    #     fourhot2[i].flat[labels24[i]] = 0.5
    # # for j in range(32):
    # #     print(fourhot[j])
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # y1 = torch.tensor(fourhot).to(device)
    # y2 = torch.tensor(fourhot2).to(device)
    # x_hot_loss = x_hot_CrossEntropy()
    # fourhot_loss_v3 = x_hot_loss(x1,y1)  + x_hot_loss(x2,y2) # 加SwAV
    # # memax损失项
    # proto_labels = one_hot(torch.tensor([i for i in range(args.cluster)]), args.cluster)
    # probs1 = snn(q_online, Memory_Bank.W, proto_labels)
    # with torch.no_grad():
    #     targets1 = snn(k_target, Memory_Bank.W, proto_labels)
    #     targets1 = sharpen(targets1, T=0.25) 
    #     targets1 = distributed_sinkhorn(targets1)
    #     targets1 = torch.cat([targets1 for _ in range(1)], dim=0)
    #     # for i in range(targets1.size(1)):
    #     #     print(targets1[0][i])
    #     targets1 = (targets1 * fourhot).cuda()
    #     targets1[targets1 < 1e-4] *= 0
    #     # for i in range(targets1.size(1)):
    #     #     print(targets1[0][i])
    # # print("*********", labels1[0], labels12[0], labels13[0], labels14[0])
    # bloss1 = torch.mean(torch.sum(torch.log(probs1**(-targets1)), dim=1))  # cross-entropy损失H(targets, queries)
    # avg_probs1 = torch.mean(probs1, dim=0)
    # rloss1 = - torch.sum(torch.log(avg_probs1**(-avg_probs1))) + math.log(float(len(avg_probs1)))  
    # probs2 = snn(k_online, Memory_Bank.W, proto_labels)
    # with torch.no_grad():
    #     targets2 = snn(q_target, Memory_Bank.W, proto_labels)
    #     targets2 = sharpen(targets2, T=0.25) 
    #     targets2 = distributed_sinkhorn(targets2)
    #     targets2 = torch.cat([targets2 for _ in range(1)], dim=0)
    #     targets2 = (targets2 * fourhot2).cuda()
    #     targets2[targets2 < 1e-4] *= 0
    # bloss2 = torch.mean(torch.sum(torch.log(probs2**(-targets2)), dim=1))  # cross-entropy损失H(targets, queries)
    # avg_probs2 = torch.mean(probs2, dim=0)
    # rloss2 = - torch.sum(torch.log(avg_probs2**(-avg_probs2))) + math.log(float(len(avg_probs2)))  
    # memax_loss = bloss1 + rloss1 + bloss2 + rloss2
    # print("v3loss + bloss1 + rloss1 + bloss2 + rloss2", fourhot_loss_v3, bloss1 , rloss1 , bloss2 , rloss2)


    # ┌─────────────────────────────────────────┐
    # |     计算4-hot v4损失+me-max损失改 
    # └─────────────────────────────────────────┘
    # x1 = logits1
    # x2 = logits2
    # fourhot = np.zeros((args.batch_size, args.cluster))
    # for i in range(labels1.size(0)):
    #     # print(labels1[i])
    #     fourhot[i] = np.zeros((1, args.cluster))
    #     fourhot[i].flat[labels1[i]] = 1
    #     fourhot[i].flat[labels12[i]] = 0.5
    #     fourhot[i].flat[labels13[i]] = 0.5
    #     # fourhot[i].flat[labels14[i]] = 0.5
    # fourhot2 = np.zeros((args.batch_size, args.cluster))
    # for i in range(labels2.size(0)):
    #     # print(labels1[i])
    #     fourhot2[i] = np.zeros((1, args.cluster))
    #     fourhot2[i].flat[labels2[i]] = 1
    #     fourhot2[i].flat[labels22[i]] = 0.5
    #     fourhot2[i].flat[labels23[i]] = 0.5
    #     # fourhot2[i].flat[labels24[i]] = 0.5
    # # for j in range(32):
    # #     print(fourhot[j])
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # y1 = torch.tensor(fourhot).to(device)
    # y2 = torch.tensor(fourhot2).to(device)
    # x_hot_loss = x_hot_CrossEntropy()
    # fourhot_loss_v4 = x_hot_loss(x1,y1)  + x_hot_loss(x2,y2) # 加SwAV
    # avg_probs = torch.mean(torch.nn.functional.normalize(logits1), dim=0)
    # rloss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))     

    # ┌─────────────────────────────────────────┐
    # |              计算3-hot 损失 
    # └─────────────────────────────────────────┘
    # x1 = logits1
    # x2 = logits2
    # threehot = np.zeros((args.batch_size, args.cluster))
    # for i in range(labels1.size(0)):
    #     # print(labels1[i])
    #     threehot[i] = np.zeros((1, args.cluster))
    #     threehot[i].flat[labels1[i]] = 1
    #     threehot[i].flat[labels12[i]] = 0.5
    #     threehot[i].flat[labels13[i]] = 0.5
    # threehot2 = np.zeros((args.batch_size, args.cluster))
    # for i in range(labels2.size(0)):
    #     # print(labels1[i])
    #     threehot2[i] = np.zeros((1, args.cluster))
    #     threehot2[i].flat[labels2[i]] = 1
    #     threehot2[i].flat[labels22[i]] = 0.5
    #     threehot2[i].flat[labels23[i]] = 0.5
    # # for j in range(32):
    # #     print(fourhot[j])
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # y1 = torch.tensor(threehot).to(device)
    # y2 = torch.tensor(threehot2).to(device)
    # x_hot_loss = x_hot_CrossEntropy()
    # threehot_loss = x_hot_loss(x1,y1)  + x_hot_loss(x2,y2) # 加SwAV



    # ┌─────────────────────────────────────────┐
    # |             计算多次单边caco损失 
    # └─────────────────────────────────────────┘
    # mpo_loss = criterion(logits1, labels1) + criterion(logits1, labels12) + criterion(logits1, labels13) + criterion(logits1, labels14)


    # if args.seed == 6:
    #     alpha = np.log(1+np.exp(memaxloss))
    # elif args.seed == 7:
    #     alpha = np.log(1+np.exp(-memaxloss))
    # elif args.seed == 8:
    #     alpha = 1 / (1 + np.exp(memaxloss))
    # elif args.seed == 9:
    #     alpha = 1 / (1 + np.exp(-memaxloss))  # better
    # else:
    #     print("未定义的种子")

    # ┌─────────────────────────────────────────┐
    # |              计算4-hot 损失 
    # └─────────────────────────────────────────┘
    # x1 = logits1
    # fourhot = np.zeros((args.batch_size, args.cluster))
    # for i in range(labels1.size(0)):
    #     if args.loss=="2hot+memax":
    #         fourhot[i].flat[labels1[i]] = 1
    #         fourhot[i].flat[labels12[i]] = 0.5
    #     if args.loss=="3hot+memax":
    #         fourhot[i].flat[labels1[i]] = 1
    #         fourhot[i].flat[labels12[i]] = 0.5
    #         fourhot[i].flat[labels13[i]] = 0.5
    #     if args.loss=="4hot+memax":
    #         fourhot[i].flat[labels1[i]] = 1
    #         fourhot[i].flat[labels12[i]] = 0.5
    #         fourhot[i].flat[labels13[i]] = 0.5
    #         fourhot[i].flat[labels14[i]] = 0.5
    #     if args.loss=="5hot+memax":
    #         fourhot[i].flat[labels1[i]] = 1
    #         fourhot[i].flat[labels12[i]] = 0.5
    #         fourhot[i].flat[labels13[i]] = 0.5
    #         fourhot[i].flat[labels14[i]] = 0.5
    #         fourhot[i].flat[labels15[i]] = 0.5
    #     if args.loss=="6hot+memax":
    #         fourhot[i].flat[labels1[i]] = 1
    #         fourhot[i].flat[labels12[i]] = 0.5
    #         fourhot[i].flat[labels13[i]] = 0.5
    #         fourhot[i].flat[labels14[i]] = 0.5
    #         fourhot[i].flat[labels15[i]] = 0.5
    #         fourhot[i].flat[labels16[i]] = 0.5
    # # for j in range(32):
    # #     print(labels1[j], labels12[j], labels13[j], labels14[j])
    # #     print(x1[j])
    # #     print(fourhot[j])
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # y1 = torch.tensor(fourhot).to(device)
    # x_hot_loss = x_hot_CrossEntropy()
    # xhot_loss = x_hot_loss(x1,y1) 

    # ┌─────────────────────────────────────────┐
    # |              计算3-hot 损失 
    # └─────────────────────────────────────────┘
    # x1b = logits1
    # threehot = np.zeros((args.batch_size, args.cluster))
    # for i in range(labels1.size(0)):
    #     threehot[i].flat[labels1[i]] = 1
    #     threehot[i].flat[labels12[i]] = 0.5
    #     threehot[i].flat[labels13[i]] = 0.5
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # y1b = torch.tensor(threehot).to(device)
    # x_hot_loss = x_hot_CrossEntropy()
    # threehot_loss = x_hot_loss(x1b,y1b) 

    # ┌─────────────────────────────────────────┐
    # |             计算4-hot全1损失 
    # └─────────────────────────────────────────┘
    # x1a = logits1
    # fourhota = np.zeros((args.batch_size, args.cluster))
    # for i in range(labels1.size(0)):
    #     fourhota[i].flat[labels1[i]] = 1
    #     fourhota[i].flat[labels12[i]] = 1
    #     fourhota[i].flat[labels13[i]] = 1
    #     fourhota[i].flat[labels14[i]] = 1
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # y1a = torch.tensor(fourhota).to(device)
    # x_hot_loss = x_hot_CrossEntropy()
    # fourhota_loss = x_hot_loss(x1a,y1a) 

    # ┌─────────────────────────────────────────┐
    # |           计算最小表征相似度损失 
    # └─────────────────────────────────────────┘
    # features = np.zeros((args.batch_size, 4))
    # for i in range(args.batch_size):
    #     features[i][0] = labels1[i]
    #     features[i][1] = labels12[i]
    #     features[i][2] = labels13[i]
    #     features[i][3] = labels14[i]
    # uniFeatures = np.unique(features)
    # repetition = 1 - len(uniFeatures)/(args.batch_size * 4)

    # ┌─────────────────────────────────────────┐
    # |           计算动态权重多hot损失msn 
    # └─────────────────────────────────────────┘
    # Dy_proto_labels = one_hot(torch.tensor([i for i in range(args.cluster)]), args.cluster)
    # Dy_probs = snn(q_online, Memory_Bank.W, Dy_proto_labels)
    # Dy_avg_probs = torch.mean(Dy_probs, dim=0)
    # Dyrloss = - torch.sum(torch.log(Dy_avg_probs**(-Dy_avg_probs))) + math.log(float(len(Dy_avg_probs)))  
    # alpha = 1 / (1 + np.exp(Dyrloss))  # 无符号是取负的
    # alpha = np.log(1+np.exp(memaxloss))
    # alpha = 1 / (1 + np.exp(memaxloss))

    # alpha = 1 / (1 + np.exp(-memaxloss))
    # Dy_x1 = logits1
    # Dy_fourhot = np.zeros((args.batch_size, args.cluster))
    # for i in range(labels1.size(0)):
    #     Dy_fourhot[i].flat[labels1[i]] = 1
    #     if args.loss=="Dy2hot+memax":
    #         Dy_fourhot[i].flat[labels12[i]] = alpha
    #     if args.loss=="Dy3hot+memax":
    #         Dy_fourhot[i].flat[labels12[i]] = alpha
    #         Dy_fourhot[i].flat[labels13[i]] = alpha
    #     if args.loss=="Dy4hot+memax" or args.loss=="0.8Dy4hot+memax":
    #         Dy_fourhot[i].flat[labels12[i]] = alpha
    #         Dy_fourhot[i].flat[labels13[i]] = alpha
    #         Dy_fourhot[i].flat[labels14[i]] = alpha
    #     if args.loss=="Dy5hot+memax":
    #         Dy_fourhot[i].flat[labels12[i]] = alpha
    #         Dy_fourhot[i].flat[labels13[i]] = alpha
    #         Dy_fourhot[i].flat[labels14[i]] = alpha
    #         Dy_fourhot[i].flat[labels15[i]] = alpha
    #     if args.loss=="Dy6hot+memax":
    #         Dy_fourhot[i].flat[labels12[i]] = alpha
    #         Dy_fourhot[i].flat[labels13[i]] = alpha
    #         Dy_fourhot[i].flat[labels14[i]] = alpha
    #         Dy_fourhot[i].flat[labels15[i]] = alpha
    #         Dy_fourhot[i].flat[labels16[i]] = alpha
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Dy_y1 = torch.tensor(Dy_fourhot).to(device)
    # x_hot_loss = x_hot_CrossEntropy()
    # Dyxhot = x_hot_loss(Dy_x1,Dy_y1) 

    # ┌─────────────────────────────────────────┐
    # |          计算动态权重公式4hot损失 
    # └─────────────────────────────────────────┘
    # Dy_x1 = logits1
    # Dy_fourhot = np.zeros((args.batch_size, args.cluster))
    # for i in range(labels1.size(0)):
    #     Dy_fourhot[i] = np.zeros((1, args.cluster))
    #     Dy_fourhot[i].flat[labels1[i]] = 1
    #     Dy_fourhot[i].flat[labels12[i]] = alpha
    #     Dy_fourhot[i].flat[labels13[i]] = alpha
    #     Dy_fourhot[i].flat[labels14[i]] = alpha
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Dy_y1 = torch.tensor(Dy_fourhot).to(device)
    # x_hot_loss = x_hot_CrossEntropy()
    # Dyxhot = x_hot_loss(Dy_x1,Dy_y1) 

    # ┌─────────────────────────────────────────┐
    # |          计算动态权重公式xhot损失 
    # └─────────────────────────────────────────┘
    # Dy_x1 = logits1
    # Dy_fourhot = np.zeros((args.batch_size, args.cluster))
    # for i in range(labels1.size(0)):
    #     if args.loss=="Dy2hot+memax" or args.loss=="2Dy2hot+memax" or args.loss=="3Dy2hot+memax" or args.loss=="4Dy2hot+memax" or args.loss=="Dy2hot+2memax" or args.loss=="Dy2hot+3memax" or args.loss=="Dy2hot+4memax":
    #         Dy_fourhot[i] = np.zeros((1, args.cluster))
    #         Dy_fourhot[i].flat[labels1[i]] = 1
    #         Dy_fourhot[i].flat[labels12[i]] = alpha
    #     elif args.loss=="Dy3hot+memax" or args.loss=="2Dy3hot+memax" or args.loss=="3Dy3hot+memax" or args.loss=="4Dy3hot+memax" or args.loss=="Dy3hot+2memax" or args.loss=="Dy3hot+3memax" or args.loss=="Dy3hot+4memax":
    #         Dy_fourhot[i] = np.zeros((1, args.cluster))
    #         Dy_fourhot[i].flat[labels1[i]] = 1
    #         Dy_fourhot[i].flat[labels12[i]] = alpha
    #         Dy_fourhot[i].flat[labels13[i]] = alpha
    #     elif args.loss=="Dy4hot+memax" or args.loss=="2Dy4hot+memax" or args.loss=="3Dy4hot+memax" or args.loss=="4Dy4hot+memax" or args.loss=="Dy4hot+2memax" or args.loss=="Dy4hot+3memax" or args.loss=="Dy4hot+4memax":
    #         Dy_fourhot[i] = np.zeros((1, args.cluster))
    #         Dy_fourhot[i].flat[labels1[i]] = 1
    #         Dy_fourhot[i].flat[labels12[i]] = alpha
    #         Dy_fourhot[i].flat[labels13[i]] = alpha
    #         Dy_fourhot[i].flat[labels14[i]] = alpha
    #     elif args.loss=="Dy5hot+memax" or args.loss=="2Dy5hot+memax" or args.loss=="3Dy5hot+memax" or args.loss=="4Dy5hot+memax" or args.loss=="Dy5hot+2memax" or args.loss=="Dy5hot+3memax" or args.loss=="Dy5hot+4memax":
    #         Dy_fourhot[i] = np.zeros((1, args.cluster))
    #         Dy_fourhot[i].flat[labels1[i]] = 1
    #         Dy_fourhot[i].flat[labels12[i]] = alpha
    #         Dy_fourhot[i].flat[labels13[i]] = alpha
    #         Dy_fourhot[i].flat[labels14[i]] = alpha
    #         Dy_fourhot[i].flat[labels15[i]] = alpha
    #     elif args.loss=="Dy6hot+memax" or args.loss=="2Dy6hot+memax" or args.loss=="3Dy6hot+memax" or args.loss=="4Dy6hot+memax" or args.loss=="Dy6hot+2memax" or args.loss=="Dy6hot+3memax" or args.loss=="Dy6hot+4memax":
    #         Dy_fourhot[i] = np.zeros((1, args.cluster))
    #         Dy_fourhot[i].flat[labels1[i]] = 1
    #         Dy_fourhot[i].flat[labels12[i]] = alpha
    #         Dy_fourhot[i].flat[labels13[i]] = alpha
    #         Dy_fourhot[i].flat[labels14[i]] = alpha
    #         Dy_fourhot[i].flat[labels15[i]] = alpha
    #         Dy_fourhot[i].flat[labels16[i]] = alpha
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Dy_y1 = torch.tensor(Dy_fourhot).to(device)
    # x_hot_loss = x_hot_CrossEntropy()
    # Dyxhot = x_hot_loss(Dy_x1,Dy_y1) 


    # ┌─────────────────────────────────────────┐
    # |             计算me-max损失 
    # └─────────────────────────────────────────┘ 
    #1
    # proto_labels = one_hot(torch.tensor([i for i in range(args.cluster)]), args.cluster)
    # probs = snn(q_pred, Memory_Bank.W, proto_labels)
    # avg_probs = AllReduce.apply(torch.mean(probs, dim=0))
    # HB =  torch.sum(torch.log(avg_probs**(-avg_probs))) 
    # # memaxloss = -HB + math.log(float(len(avg_probs)))  
    # memaxloss = -HB + math.log(args.cluster,2)
    # alpha = 1 / (1 + np.exp(-(-memaxloss)))
    #2
    # proto_labels = one_hot(torch.tensor([i for i in range(args.cluster)]), args.cluster)
    # probs = snn(q_pred, Memory_Bank.W, proto_labels)
    # avg_probs = AllReduce.apply(torch.mean(probs, dim=0))
    # HB =  torch.sum(torch.log(avg_probs**(-avg_probs))) 
    # memaxloss = -HB + math.log(float(len(avg_probs)))  
    # alpha = 1 / (1 + np.exp(-memaxloss))
    #3
    # avg_probs = torch.mean(torch.nn.functional.normalize(logits1), dim=0)  # 95.56 +-0.03 95.00
    # HB =  torch.sum(torch.log(avg_probs**(-avg_probs)))
    # memaxloss = -HB + math.log(float(len(avg_probs)))
    # alpha = 1 / (1 + np.exp(-memaxloss))

    # avg_probs = torch.nn.functional.normalize(torch.mean(logits1, dim=0), dim=0)  # 93.26 +-0.01 93.25
    # avg_probs = torch.nn.functional.normalize(torch.mean(torch.nn.functional.normalize(logits1), dim=0), dim=0)  # 92.95 +-0.01
    # softmax = torch.nn.Softmax(dim=1)
    # avg_probs = torch.nn.functional.normalize(torch.mean(softmax(logits1), dim=0), dim=0)  # 82.92
    
    #4 跑着跑着loss出现nan,给memaxloss赋6倍权重后91.88 +-0.04
    # avg_probs = torch.nn.functional.normalize(torch.mean(logits1, dim=0), p=1, dim=0)
    # HB = torch.sum(torch.log(avg_probs**(-avg_probs)))
    # memaxloss = -HB + math.log(float(len(avg_probs)))
    # alpha = 1 / (1 + np.exp(-(HB-math.log(float(len(avg_probs))))))
    # alpha = 0.5
    # alpha = 1 / (1 + np.exp(HB-math.log(float(len(avg_probs)))))
    #5 
    # class MEMAXRegularizer(torch.nn.Module):
    #     def __init__(self, threshold=0.5):
    #         super(MEMAXRegularizer, self).__init__()
    #         self.threshold = threshold
        
    #     def forward(self, logits):
    #         probs = F.sigmoid(logits)
    #         entropies = -probs * torch.log(probs) - (1 - probs) * torch.log(1 - probs)
    #         mean_entropy = torch.mean(entropies, dim=1)
    #         max_entropy, _ = torch.max(entropies, dim=1)
    #         regularizer_loss = torch.mean(torch.clamp(max_entropy - mean_entropy + self.threshold, min=0))
    #         return regularizer_loss
    # memax_regularizer = MEMAXRegularizer(threshold=0.5)
    # memaxloss = memax_regularizer(logits1)
    # # alpha = 1 / (1 + np.exp(-(-memaxloss)))  # 88.49 +-0.00
    # alpha = 1 / (1 + np.exp((-memaxloss)))
    #6
    # avg_probs1 = torch.mean(torch.nn.functional.normalize(logits1), dim=0)  # 95.56 +-0.03 95.00
    # avg_probs2 = torch.mean(torch.nn.functional.normalize(logits2), dim=0)
    # memaxloss1 = - torch.sum(torch.log(avg_probs1**(-avg_probs1))) + math.log(float(len(avg_probs1)))
    # memaxloss2 = - torch.sum(torch.log(avg_probs2**(-avg_probs2))) + math.log(float(len(avg_probs2)))
    # memaxloss = (memaxloss1 + memaxloss2) * 0.5
    # alpha = 1 / (1 + np.exp(-memaxloss))
    #7
    # proto_labels = one_hot(torch.tensor([i for i in range(args.cluster)]), args.cluster)
    # probs = snn(q_pred, Memory_Bank.W, proto_labels)
    # avg_probs = AllReduce.apply(torch.mean(probs, dim=0))
    # HB =  torch.sum(torch.log(avg_probs**(-avg_probs))) 
    # logK = math.log(args.cluster,2)
    # memaxloss = -HB
    # alitem = -HB + logK
    # alpha = 1 / (1 + np.exp(-(-alitem)))
    #8
    # rho = 0.4
    # rho_hat = torch.mean(torch.nn.functional.normalize(logits1), dim=0)
    # KL = rho * torch.log(rho / rho_hat) + (1-rho) * torch.log((1-rho)/(1-rho_hat))
    # KL = rho_hat * torch.log(rho_hat / rho) + (1-rho_hat) * torch.log((1-rho_hat)/(1-rho))
    # KL_loss =  torch.sum(KL) #+ math.log(float(len(rho_hat)))
    # KL_alpha = 1 / (1 + np.exp(KL_loss))  
    # KL_alpha = (1 / (1 + np.exp(KL_loss))) * (1 / (1 + np.exp(KL_loss))) * (1 / (1 + np.exp(KL_loss))) 
    # if args.loss=="Dy4hot2P+2KL":
    #     KL_alpha = (1 / (1 + np.exp(KL_loss))) * (1 / (1 + np.exp(KL_loss)))
    # if args.loss=="Dy4hot3P+3KL":
    #     KL_alpha = (1 / (1 + np.exp(KL_loss))) * (1 / (1 + np.exp(KL_loss))) * (1 / (1 + np.exp(KL_loss))) 
    # lamda = xx
    # KL_alpha = lamda / (1 + np.exp(KL_loss))  
    #9
    # rho = 0.4
    # rho_hat = torch.mean(torch.nn.functional.sigmoid(logits1), dim=0)
    # KL = rho * torch.log(rho / rho_hat) + (1-rho) * torch.log((1-rho)/(1-rho_hat))
    # KL_loss =  torch.sum(torch.log(KL)) + math.log(float(len(rho_hat)))
    # KL_alpha = 1 / (1 + np.exp(-KL_loss))
    #10
    # rho = 0.4
    # rho_hat = torch.mean(torch.nn.functional.normalize(logits1), dim=0)
    # KL = rho_hat * torch.log(rho_hat / rho) + (1-rho_hat) * torch.log((1-rho_hat)/(1-rho))
    # KL_loss =  torch.sum(KL) 
    # KL_alpha = 1 / (1 + np.exp(KL_loss))  