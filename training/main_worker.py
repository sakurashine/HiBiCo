import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import math
import model.ResNet as models
from model.CaCo import CaCo, CaCo_PN
from ops.os_operation import mkdir_rank
from training.train_utils import adjust_learning_rate2,save_checkpoint
from data_processing.datasets import get_dataset,Hyper2X
import warnings
warnings.filterwarnings("ignore", category=Warning)


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


def main_worker(gpu, args):
    params = vars(args)
    args.gpu = gpu
    init_lr = args.lr
    total_batch_size = args.batch_size
    # suppress printing if not master
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    seed = args.seed 
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
    if args.dataset=='IndianPines' or args.dataset=='Botswana' or args.dataset=='HyRANK' or args.dataset=='Houston18':
        DATASET = args.dataset
        FOLDER = args.data_folder
        hyperparams = vars(args)
        hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)
        img, gt, LABEL_VALUES, IGNORED_LABELS = get_dataset(DATASET, FOLDER)
        N_CLASSES = len(LABEL_VALUES)
        N_BANDS = img.shape[-1]
        hyperparams.update(
            {'n_classes': N_CLASSES, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS})
        train_gt = gt
        mask = np.unique(train_gt)
        tmp = []
        for v in mask:
            tmp.append(np.sum(train_gt==v))
        print("类别：{}".format(mask))
        print("训练集每类个数{}".format(tmp))
        print("挑选{}个训练样本，总计{}个可用样本".format(np.count_nonzero(train_gt), np.count_nonzero(gt)))
        train_dataset = Hyper2X(img, train_gt, **hyperparams)
        print(args.dataset + "数据集加载完毕!")

    else:
        print("We don't support this dataset currently")
        exit()


    Memory_Bank = CaCo_PN(args.cluster,args.moco_dim)
    LAYER = args.layer
    layer_config = {
                    1: 28800,
                    2: 28800,
                    3: 8192,
                    15: 28800,
                    25: 8192
                    }
    if LAYER in layer_config:
        Memory_Bank_upper = CaCo_PN(args.cluster_upper, layer_config[LAYER])
    else:
        print("Please choose which layer to construct shallow dictionary!")
    
    model = CaCo(models.__dict__[args.arch], args, args.moco_dim, args.moco_m, N_BANDS, LAYER)
    from model.optimizer import  LARS
    optimizer = LARS(model.parameters(), init_lr,
                         weight_decay=args.weight_decay,
                         momentum=args.momentum)
    
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        Memory_Bank=Memory_Bank.cuda(args.gpu)
        Memory_Bank_upper=Memory_Bank_upper.cuda(args.gpu)
    else:
        model.cuda()
        Memory_Bank.cuda()
    
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    save_path = init_log_path(args,total_batch_size)
    print("save_path: ", save_path)
    if not args.resume:
        args.resume = os.path.join(save_path,"checkpoint_best.pth.tar")
        print("searching resume files ",args.resume)

    cudnn.benchmark = True
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    model.eval()
    if args.ad_init and not os.path.isfile(args.resume):
        from training.init_memory import init_memory
        init_memory(train_loader, model, Memory_Bank, Memory_Bank_upper, criterion,
                optimizer, 0, args)
    
    train_log_path = os.path.join(save_path,"train.log")
    best_Acc=0
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate2(optimizer, epoch, args, init_lr)    
        if args.moco_m_decay:
            moco_momentum = adjust_moco_momentum(epoch, args)
        else:
            moco_momentum = args.moco_m
        from training.train_caco import train_caco
        acc1 = train_caco(train_loader, model, Memory_Bank, Memory_Bank_upper, criterion,
                                optimizer, epoch, args, train_log_path,moco_momentum)  
        is_best=best_Acc>acc1
        best_Acc=max(best_Acc,acc1)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank == 0):
            save_dict={
            'epoch': epoch + 1,
            'arch': args.arch,
            'best_acc':best_Acc,
            'knn_acc': 0,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'Memory_Bank':Memory_Bank.state_dict(),
            }
            if epoch%10==9:
                tmp_save_path = os.path.join(save_path, 'checkpoint_{:04d}.pth.tar'.format(epoch))
                save_checkpoint(save_dict, is_best=False, filename=tmp_save_path)
            tmp_save_path = os.path.join(save_path, 'checkpoint_best.pth.tar')
            save_checkpoint(save_dict, is_best=is_best, filename=tmp_save_path)
           
        
def adjust_moco_momentum(epoch, args):
    """Adjust moco momentum based on current epoch"""
    return 1. - 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (1. - args.moco_m)
