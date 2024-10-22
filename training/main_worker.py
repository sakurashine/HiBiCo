
import builtins
import torch.distributed as dist
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import datetime
import time
import numpy as np
import math

import model.ResNet as models
from model.CaCo import CaCo, CaCo_PN
from ops.os_operation import mkdir, mkdir_rank
from training.train_utils import adjust_learning_rate2,save_checkpoint,pca_dr, applyPCA
from data_processing.loader import TwoCropsTransform, TwoCropsTransform2,GaussianBlur,Solarize
from ops.knn_monitor import knn_monitor
from data_processing import tSNE
from data_processing.IndianPines import get_dataset,Hyper2X,HyperX
from data_processing.utils import sample_gt, is_dist_avail_and_initialized, get_rank
# from loss import dcl
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore", category=Warning)


# 按多个参数分子文件夹存储模型文件和log文件
def init_log_path(args,batch_size):
    save_path = os.path.join(os.getcwd(), args.log_path)
    mkdir_rank(save_path,args.rank)
    save_path = os.path.join(save_path, args.dataset)
    mkdir_rank(save_path,args.rank)
    save_path = os.path.join(save_path, "Type_"+str(args.type))
    mkdir_rank(save_path,args.rank)
    save_path = os.path.join(save_path, "lr_" + str(args.lr) + "_" + str(args.lr_final))
    mkdir_rank(save_path,args.rank)
    save_path = os.path.join(save_path, "memlr_"+str(args.memory_lr) +"_"+ str(args.memory_lr_final))
    mkdir_rank(save_path,args.rank)
    save_path = os.path.join(save_path, "t_" + str(args.moco_t) + "_memt" + str(args.mem_t))
    mkdir_rank(save_path,args.rank)
    save_path = os.path.join(save_path, "wd_" + str(args.weight_decay) + "_memwd" + str(args.mem_wd)) 
    mkdir_rank(save_path,args.rank)
    if args.moco_m_decay:
        save_path = os.path.join(save_path, "mocomdecay_" + str(args.moco_m))
    else:
        save_path = os.path.join(save_path, "mocom_" + str(args.moco_m))
    mkdir_rank(save_path,args.rank)
    save_path = os.path.join(save_path, "memgradm_" + str(args.mem_momentum))
    mkdir_rank(save_path,args.rank)
    save_path = os.path.join(save_path, "hidden" + str(args.mlp_dim)+"_out"+str(args.moco_dim))
    mkdir_rank(save_path,args.rank)
    save_path = os.path.join(save_path, "batch_" + str(batch_size))
    mkdir_rank(save_path,args.rank)
    save_path = os.path.join(save_path, "epoch_" + str(args.epochs))
    mkdir_rank(save_path,args.rank)
    save_path = os.path.join(save_path, "warm_" + str(args.warmup_epochs))
    mkdir_rank(save_path,args.rank)
    save_path = os.path.join(save_path, "time_" + str(args.time))
    mkdir_rank(save_path,args.rank)
    return save_path

# 主线程
def main_worker(gpu, ngpus_per_node, args):
    params = vars(args)
    args.gpu = gpu
    # 打印所有超参数
    print(vars(args))
    # 定义初始学习率，超参数设定的学习率在此经过计算后才是真正的初始学习率，batch取32的话就相当于设定值除以8
    init_lr = args.lr
    # init_lr = args.lr * args.batch_size / 256  # batch取32时相当于除以8
    print("设定学习率", args.lr)
    print("初始学习率", init_lr)
    total_batch_size = args.batch_size
    #args.memory_lr = args.memory_lr * args.batch_size / 256
    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        # 未进入此条件
        if args.dist_url == "env://" and args.rank == -1:
                args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
                # For multiprocessing distributed training, rank needs to be the
                # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu#we only need to give rank to this machine, then it's enough
            #world size in multi node specified the number of total nodes
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        
        if args.nodes_num>1:
            args.batch_size = args.batch_size // args.nodes_num
            args.knn_batch_size = args.knn_batch_size // args.nodes_num
            args.workers = args.workers//args.nodes_num
            torch.distributed.barrier()
    
    # 添加随机种子
    seed = args.seed 
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print("打印随机种子：", np.random.seed(seed))

    # 打印rank，rank标记第几个进程
    print("rank ",args.rank)
    # create model ResNet50
    print("=> creating model '{}'".format(args.arch))

    # 数据集加载
    print("打印args")
    print(vars(args))
    print(args.dataset)
    
    if args.dataset=='imagenet-mini':  
        traindir = os.path.join(args.data_folder, 'train')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        if args.multi_crop:
            from data_processing.MultiCrop_Transform import Multi_Transform
            multi_transform = Multi_Transform(args.size_crops,
                                              args.nmb_crops,
                                              args.min_scale_crops,
                                              args.max_scale_crops, normalize)
            train_dataset = datasets.ImageFolder(
                traindir, multi_transform)
        else:

            augmentation1 = [
                    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ]

            augmentation2 = [
                    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
                    transforms.RandomApply([Solarize()], p=0.2),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ]
            train_dataset = datasets.ImageFolder(
                    traindir,
                    TwoCropsTransform2(transforms.Compose(augmentation1),
                                       transforms.Compose(augmentation2)))
            print("打印type(train_dataset)",type(train_dataset))  # <class 'torchvision.datasets.folder.ImageFolder'>
        testdir = os.path.join(args.data, 'val')
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        from data_processing.imagenet import imagenet
        val_dataset = imagenet(traindir, 0.2, transform_test)
        test_dataset = datasets.ImageFolder(testdir, transform_test)
        print("imagenet-mini数据集加载完毕")
    
    elif args.dataset=='cifar10':  
        traindir = os.path.join(args.data_folder, 'train')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        if args.multi_crop:
            from data_processing.MultiCrop_Transform import Multi_Transform
            multi_transform = Multi_Transform(args.size_crops,
                                              args.nmb_crops,
                                              args.min_scale_crops,
                                              args.max_scale_crops, normalize)
            train_dataset = datasets.ImageFolder(
                traindir, multi_transform)
        else:

            augmentation1 = [
                    # transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ]

            augmentation2 = [
                    # transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
                    transforms.RandomApply([Solarize()], p=0.2),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ]
            train_dataset = datasets.ImageFolder(
                    traindir,
                    TwoCropsTransform2(transforms.Compose(augmentation1),
                                       transforms.Compose(augmentation2)))
            print("打印type(train_dataset)",type(train_dataset))  # <class 'torchvision.datasets.folder.ImageFolder'>
        testdir = os.path.join(args.data_folder, 'val')
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        from data_processing.cifar10 import cifar10
        val_dataset = cifar10(traindir, 0.2, transform_test)
        test_dataset = datasets.ImageFolder(testdir, transform_test)
        print("cifar10数据集加载完毕")
    
    #//////////////////////////高光谱数据加载开始////////////////////////////////
    elif args.dataset=='IndianPines' or args.dataset=='PaviaU' or args.dataset=='WHU-Hi-HanChuan' \
        or args.dataset=='Salinas' or args.dataset=='Botswana' or args.dataset=='KSC' or args.dataset=='Houston' \
        or args.dataset=='HyRANK' or args.dataset=='Houston18' or args.dataset=='Chikusei' \
        or args.dataset=='CamPha' or args.dataset=='MongCai':
        print("使用" + args.dataset + "数据集！")
        # 数据集
        DATASET = args.dataset
        # 读取文件的目录
        FOLDER = args.data_folder
        # 加载数据的比例
        LOAD_DATA = args.load_data
        # 样本取样策略，一般为随机取样
        SAMPLING_MODE = args.sampling_mode
        # 类别平衡
        CLASS_BALANCING = args.class_balancing
        hyperparams = vars(args)
        print("dataset,folder",DATASET, FOLDER)
        print('打印参数字典:',hyperparams)
        hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)
        img, gt, LABEL_VALUES, IGNORED_LABELS = get_dataset(DATASET, FOLDER)
        # 分类数
        N_CLASSES = len(LABEL_VALUES)
        # 光谱波段数
        N_BANDS = img.shape[-1]

        # 对img进行无降维的PCA
        # img = applyPCA(img,numComponents=N_BANDS)
        # img = pca_dr(img,LOAD_DATA,DATASET,3)


        hyperparams.update(
            {'n_classes': N_CLASSES, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS})
        print("训练集加载比例：",LOAD_DATA)
        if LOAD_DATA:
            train_gt_file = '../dataset/' + DATASET + '/' + LOAD_DATA + '/train_gt.npy'
            test_gt_file  = '../dataset/' + DATASET + '/' + LOAD_DATA + '/test_gt.npy'
            print("训练集Ground truth加载路径：",train_gt_file)
            # r是只读
            train_gt = np.load(train_gt_file, 'r')
            test_gt = np.load(test_gt_file, 'r')
        else:
            print("已加载全比例训练样本！")
            # train_gt, _= sample_gt(gt, 0.1, mode='random')
            train_gt = gt
        # unique去重
        mask = np.unique(train_gt)
        tmp = []
        for v in mask:
            tmp.append(np.sum(train_gt==v))
        print("类别：{}".format(mask))
        print("训练集每类个数{}".format(tmp))
        print("挑选{}个训练样本，总计{}个可用样本".format(np.count_nonzero(train_gt), np.count_nonzero(gt)))
        # 此时的validation_percentage在argparser处初始化为了0.01
        print('验证集比例：',hyperparams['validation_percentage'])
        val_gt, _ = sample_gt(gt, train_size=hyperparams['validation_percentage'], mode=SAMPLING_MODE)
        test_gt, _ = sample_gt(gt, train_size=hyperparams['validation_percentage'], mode=SAMPLING_MODE)
        # display_goundtruth(gt=gt, vis=vis, caption = "Training {} samples selected".format(np.count_nonzero(gt)))

        # 类别平衡，权重值为该类频数占所有类别频数中位数的倍数，比如第一类松树像素个数是第五类的十四倍
        #   weights:[ 0.         14.          0.3255814   0.56        2.          1.
        #             0.63636364 14.          1.          14.         0.48275862  0.19178082
        #             0.77777778 2.33333333   0.36842105  1.16666667  4.66666667 ]
        # if CLASS_BALANCING:
        #     weights = compute_imf_weights(train_gt, N_CLASSES, IGNORED_LABELS)
            # print('开始打印类别平衡weights',weights)
            #    hyperparams.update({'weights': torch.from_numpy(weights)})
            # hyperparams['weights'] = torch.from_numpy(weights).float().cuda()
        
        train_dataset = Hyper2X(img, train_gt, **hyperparams)
        val_dataset = Hyper2X(img, val_gt, **hyperparams)
        test_dataset = HyperX(img, test_gt, **hyperparams)

        print(args.dataset + "数据集加载完毕!")
        #//////////////////////////高光谱数据加载完毕////////////////////////////////

    else:
        print("We don't support this dataset currently")
        exit()

    # 新建Memory Bank，同时新建浅层字典
    Memory_Bank = CaCo_PN(args.cluster,args.moco_dim)
    # Memory_Bank_upper = CaCo_PN(args.cluster_upper,args.moco_dim)  # 匹配conv1表征的维度

     # 判断字典构建层，并决定浅层字典大小
    LAYER = args.layer
    if LAYER == 1 or LAYER == 2:
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXX", LAYER)
        Memory_Bank_upper = CaCo_PN(args.cluster_upper,28800)  # 匹配conv1表征的维度
    elif LAYER == 3:
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXX", LAYER)
        Memory_Bank_upper = CaCo_PN(args.cluster_upper,8192)  # 匹配conv1表征的维度
    elif LAYER == 15:
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXX", LAYER)
        Memory_Bank_upper = CaCo_PN(args.cluster_upper,28800)  # 匹配conv1表征的维度
    elif LAYER == 25:
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXX", LAYER)
        Memory_Bank_upper = CaCo_PN(args.cluster_upper,8192)  # 匹配conv1表征的维度
    else:
        print("Choose which layer to construct upper dictionary!")
    
    # 新建model
    model = CaCo(models.__dict__[args.arch], args, args.moco_dim, args.moco_m, N_BANDS, LAYER)  # 传N_BANDS避免了更换数据集时需要到ResNet.py改波段数
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)  # 类似全局BN

    # 即使更换了优化器，依然对准确率没什么影响
    from model.optimizer import  LARS
    optimizer = LARS(model.parameters(), init_lr,
                         weight_decay=args.weight_decay,
                         momentum=args.momentum)
    # optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=0.0001)
    if args.distributed:
        # 未进入此条件
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            args.knn_batch_size =int(args.knn_batch_size / ngpus_per_node) 
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=True)
            Memory_Bank = Memory_Bank.cuda(args.gpu)
            Memory_Bank_upper = Memory_Bank.cuda(args.gpu)
        else:
            model.cuda()
            Memory_Bank.cuda()
            Memory_Bank_upper.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model,find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        Memory_Bank=Memory_Bank.cuda(args.gpu)
        Memory_Bank_upper=Memory_Bank_upper.cuda(args.gpu)
        # comment out the following line for debugging
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        model.cuda()
        Memory_Bank.cuda()
        print("Only DistributedDataParallel is supported.")
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    
    print("per gpu batch size: ",args.batch_size)
    print("current workers:",args.workers)
    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    # 此处可做修改
    # criterion = ContrastiveLoss()
    # criterion = dcl.DCL(temperature=0.5)
    

    save_path = init_log_path(args,total_batch_size)
    print("save_path: ", save_path)
    if not args.resume:
        print("进入if not args.resume条件")
        args.resume = os.path.join(save_path,"checkpoint_best.pth.tar")
        print("searching resume files ",args.resume)
    # optionally resume from a checkpoint
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         print("=> loading checkpoint '{}'".format(args.resume))
    #         if args.gpu is None:
    #             checkpoint = torch.load(args.resume)
    #         else:
    #             # Map model to be loaded to specified single gpu.
    #             loc = 'cuda:{}'.format(args.gpu)
    #             checkpoint = torch.load(args.resume, map_location=loc)
    #         args.start_epoch = checkpoint['epoch']
    #         model.load_state_dict(checkpoint['state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         Memory_Bank.load_state_dict(checkpoint['Memory_Bank'])
    #         print("=> loaded checkpoint '{}' (epoch {})"
    #               .format(args.resume, checkpoint['epoch']))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None

    # 数据加载器
    # DataLoader是PyTorch中数据读取的一个重要接口，定义在dataloader.py中，只要是用PyTorch来训练模型基本都会用到该接口
    # 该接口的目的：将自定义的Dataset根据batch size大小、是否shuffle等封装成一个Batch Size大小的Tensor用于后面的训练
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.knn_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.knn_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=test_sampler, drop_last=False)


    # 初始化Memory Bank权重
    bank_size=args.cluster  # 聚类中心个数即Memory Bank的大小
    print("DataLoader配置完成!")
    model.eval()
    print("model:", model)
    
    print("运行前的GPU消耗:", torch.cuda.memory_allocated()/1024/1024)
    
    # 无字典初始化消融实验76 77 78，结论为caco57的精度，caco+KL、Dy4hot+KL无法训练
    # 初始化 memory bank
    if args.ad_init and not os.path.isfile(args.resume):
        from training.init_memory import init_memory
        init_memory(train_loader, model, Memory_Bank, Memory_Bank_upper, criterion,
                optimizer, 0, args)
        print("初始化memory bank完成!")
    
    knn_path = os.path.join(save_path,"knn.log")
    train_log_path = os.path.join(save_path,"train.log")
    best_Acc=0
    
    MemaxLoss = []
    Alpha = []
    Label = []
    # 训练阶段，遍历所有epoch
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        #adjust_learning_rate(optimizer, epoch, args)
        adjust_learning_rate2(optimizer, epoch, args, init_lr)    
        #if args.type<10:
        if args.moco_m_decay:
            moco_momentum = adjust_moco_momentum(epoch, args)
        else:
            moco_momentum = args.moco_m
        # print("当前moco momentum值 %f"%moco_momentum)
        # 训练一个epoch
        from training.train_caco import train_caco
        # print("当前epoch:", epoch)
        
        acc1, memax, alpha, Labels = train_caco(train_loader, model, Memory_Bank, Memory_Bank_upper, criterion,
                                optimizer, epoch, args, train_log_path,moco_momentum)  
        # print("logits1.shape: ", logits1.shape)  # torch.Size([32, 1024])  # 这里取到的logits应该不是memory bank存的表征，而是经过计算的
        # print("logits2.shape: ", logits2.shape)  # torch.Size([32, 1024])
        # print("第" + str(epoch) + "轮训练结束！")
        MemaxLoss.append(memax)
        Alpha.append(alpha)
        Label.append(Labels)
        '''
        # 对当轮更新的Memory Bank进行t-SNE可视化
        if(epoch %5 == 0):
            print("当前对第" + str(epoch) + "个epoch的Memory Bank进行可视化！")
            X = feature_bank[200:600,:]
            Y = feature_labels[200:600]
            digits_final = TSNE(perplexity=30).fit_transform(X) 
            plot(digits_final, Y, epoch)
        
        # 对每若干轮embedding进行t-SNE可视化
        if((epoch) %1 == 0):
            print("当前对第" + str(epoch) + "个epoch进行可视化！")
            X = feature_bank[1:,:]
            Y = feature_labels[1:]
            print("feature_bank.shape", X.shape)  # torch.Size([81409, 64])
            print("feature_labels.shape", Y.shape)  # torch.Size([81409])
            digits_final = TSNE(perplexity=30).fit_transform(X) 
            plot(digits_final, Y, epoch, args.dataset)  
            # print("打印memory bank的相关性矩阵！")
            # print("memory_bank.shape", Memory_Bank.W.shape, Memory_Bank.W)
            # corr = np.corrcoef(Memory_Bank.W)
            # print("corrcoef of memory bank", corr.shape, corr)
            # sns.set(font_scale=1.25)#字符大小设定
            # hm=sns.heatmap(corr, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
            plt.savefig('./t-SNE/' + str(datetime.date.today()) +args.dataset +'_epoch'+ str(epoch) +'_tsne.png', dpi=120)
        '''
        
        
        
        # KNN分类预测
        # 原文为epoch<=20
        '''
        if epoch%args.knn_freq==0 or epoch<=200:
            print("gpu consuming before cleaning:", torch.cuda.memory_allocated()/1024/1024)
            torch.cuda.empty_cache()
            print("gpu consuming after cleaning:", torch.cuda.memory_allocated()/1024/1024)

            # try:
            # print("打印model.module.encoder_q", model.module.encoder_q)
            knn_test_acc=knn_monitor(model.encoder_q, val_loader, test_loader,
                    global_k=min(args.knn_neighbor,len(val_loader.dataset)))
            print({'*KNN monitor Accuracy': knn_test_acc})
            if args.rank ==0:
                with open(knn_path,'a+') as file:
                    file.write('%d epoch KNN monitor Accuracy %f\n'%(epoch,knn_test_acc))
            # except:
            #     print("small error raised in knn calcu")
            #     knn_test_acc=0
            torch.cuda.empty_cache()
            epoch_limit=200
            if knn_test_acc<=1.0 and epoch>=epoch_limit:
                exit()
        '''
        is_best=best_Acc>acc1
        best_Acc=max(best_Acc,acc1)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank == 0):
            save_dict={
            'epoch': epoch + 1,
            'arch': args.arch,
            'best_acc':best_Acc,
            # 'knn_acc': knn_test_acc,
            'knn_acc': 3.14,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'Memory_Bank':Memory_Bank.state_dict(),
            }

            # checkpoint_bes与model_best的区别是？
            # if epoch%10==9:
            #     tmp_save_path = os.path.join(save_path, 'checkpoint_{:04d}.pth.tar'.format(epoch))
            #     save_checkpoint(save_dict, is_best=False, filename=tmp_save_path)
            tmp_save_path = os.path.join(save_path, 'checkpoint_best.pth.tar')

            save_checkpoint(save_dict, is_best=is_best, filename=tmp_save_path)
    # np.save('./plot/seed' + str(args.seed) + str(args.dataset) + '_time'+ str(args.time) + '_MemaxLoss.npy', MemaxLoss) # 保存为.npy格式
    # np.save('./plot/seed' + str(args.seed) + str(args.dataset) + '_time'+ str(args.time) + '_Alpha.npy', Alpha)
    # np.save('./plot/seed' + str(args.seed) + str(args.dataset) + '_time'+ str(args.time) + '_Label.npy', Label)
           
        
          
def adjust_moco_momentum(epoch, args):
    """Adjust moco momentum based on current epoch"""
    return 1. - 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (1. - args.moco_m)

# 定义t-SNE绘图函数
def plot(x, colors, epoch, datasetname):
    palette = np.array(sns.color_palette("pastel", 17))
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int8)])
    # sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.cpu().numpy().astype(np.int8)])
    txts = []
    for i in range(17):
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([pe.Stroke(linewidth=5, foreground="w"), pe.Normal()])
        txts.append(txt)
    plt.savefig('./t-SNE/' + str(datetime.date.today()) +datasetname +'_epoch'+ str(epoch) +'_tsne.png', dpi=120)
    return f, ax, txts

class ContrastiveLoss(nn.Module):

    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    def forward(self, output1, output2):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean(torch.pow(euclidean_distance, 2))

        return loss_contrastive