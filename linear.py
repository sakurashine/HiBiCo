
import argparse
import math
import os
import random
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import model.ResNet_linear as models
import numpy as np
from data_processing.datasets import get_dataset, HyperX
from data_processing.utils import  get_device, sample_gt, count_sliding_window, metrics, logger, display_goundtruth, sliding_window, grouper
from ops.LARS import SGD_LARC
from tqdm import tqdm
import matplotlib.pyplot as plt
import datetime
from torchsummary import summary
import logging


parser = argparse.ArgumentParser(description='PyTorch HSI Training')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=2, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--cuda', type=int, default=0,
                    help="Specify CUDA device (defaults to -1, which learns on CPU)")   
parser.add_argument('--test_stride', type=int, default=1,
                         help="Sliding window step stride during inference (default = 1)")  
parser.add_argument('--seed', default=16, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--pretrained', default='', type=str,
                    help='path of pretrained checkpoint')
parser.add_argument("--dataset", type=str, default="IndianPines", help="which dataset is used to test")
parser.add_argument('--folder', default="../dataset/IndianPines", type=str, metavar='DIR',
                        help='path to dataset')
parser.add_argument('--training_percentage', type=float, default=0.03,
                           help="Percentage of samples to use for training")
parser.add_argument('--sample_nums', type=int, default=None,
                           help="Number of samples to use for training, should be no larger than the size of the fewest category") 
parser.add_argument('--patch_size', type=int, default=15,
                         help="Size of the spatial neighbourhood (optional, if "
                              "absent will be set by the model)")  
parser.add_argument('--supervision', type=str, default='full',
                         help="full supervision or semi supervision ") 
parser.add_argument('--run', type=int, default=10,
                    help="Running times.")
parser.add_argument('--fine_tune', type=str, default='no',
                         help="Choose linear prob or fine-tune")    
parser.add_argument('--desc', type=str, default='1_HiBiCo',
                         help="Describing current experiment with one word")                   


def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu
    RUN = args.run
    DATASET = args.dataset
    file_date = datetime.datetime.now().strftime('%Y-%m-%d')
    log_date = datetime.datetime.now().strftime('%Y-%m-%d:%H:%M')
    log = logger('./test_log/logs-' + file_date + DATASET +'.txt')
    logging.getLogger('matplotlib.font_manager').disabled = True
    log.info("---------------------------------------------------------------------")
    log.info("-----------------------------Next run log----------------------------")
    log.info("---------------------------{}--------------------------".format(log_date))
    log.info("---------------------------------------------------------------------")
    CUDA_DEVICE = get_device(log, args.cuda)
    FOLDER = args.folder
    SAMPLE_NUMS = args.sample_nums
    TRAINING_PERCENTAGE = args.training_percentage
    SAMPLE_NUMS = args.sample_nums
    hyperparams = vars(args)
    img, gt, LABEL_VALUES, IGNORED_LABELS = get_dataset(DATASET, FOLDER)
    N_CLASSES = len(LABEL_VALUES)
    N_BANDS = img.shape[-1]
    FINE_TUNE = args.fine_tune
    hyperparams.update(
        {'n_classes': N_CLASSES, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS, 'center_pixel': True, 'device': CUDA_DEVICE, 'fine_tune': FINE_TUNE})
    hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)
    log.info("已加载{}数据集".format(DATASET))
    log.info("标签类名：{}".format(LABEL_VALUES))
    log.info("标签数量：{}".format(N_CLASSES))
    log.info("波段数：{}".format(N_BANDS))
    
    acc_dataset = np.zeros([RUN, 1])
    A = np.zeros([RUN, N_CLASSES-1])
    K = np.zeros([RUN, 1])
    
    # 默认跑10次取均值
    for i in range(RUN):
        log.info("==========================================================================================")
        log.info("======================================RUN:{}===============================================".format(i))
        log.info("==========================================================================================")
        model = models.resnet18(num_classes=N_CLASSES, num_bands=N_BANDS, fine_tune = FINE_TUNE)
        # 冻结除最后FC层外的所有层，以验证预训练模型的特征学习能力，实现linear probing而非finetune
        for name, param in model.named_parameters():
            ft = False if FINE_TUNE == 'no' else True
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = ft
        model.fc.weight.data.normal_(mean=0.0, std=0.01)
        model.fc.bias.data.zero_()
        linear_keyword="fc"
        
        # 加载预训练模型
        if args.pretrained:
            if os.path.isfile(args.pretrained): 
                print("=> loading checkpoint '{}'".format(args.pretrained))
                checkpoint = torch.load(args.pretrained, map_location="cpu")
                state_dict = checkpoint['state_dict']
                for k in list(state_dict.keys()):
                    if k.startswith('encoder_q') and not k.startswith('encoder_q.fc'):
                        state_dict[k[len("encoder_q."):]] = state_dict[k]
                        del state_dict[k]
                args.start_epoch = 0
                msg = model.load_state_dict(state_dict, strict=False)
                assert set(msg.missing_keys) == {"%s.weight" % linear_keyword, "%s.bias" % linear_keyword}
                print("=> loaded pre-trained model '{}'".format(args.pretrained))
            else:
                print("=> no checkpoint found at '{}'".format(args.pretrained))  

        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        init_lr = args.lr
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
        log.info("=> 使用 LARS 优化器")
        optimizer = torch.optim.SGD(parameters, init_lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        optimizer = SGD_LARC(optimizer, trust_coefficient=0.001, clip=False, eps=1e-8)
        cudnn.benchmark = True

        # 数据集加载   
        if SAMPLE_NUMS:
            log.info("采样方式：固定样本个数")
            train_gt, test_gt = sample_gt(gt, TRAINING_PERCENTAGE, mode='fixed', sample_nums=SAMPLE_NUMS)
            log.info("挑选{}个训练样本，总计{}个可用样本".format(np.count_nonzero(train_gt), np.count_nonzero(gt)))
            log.info("挑选{}个测试样本，总计{}个可用样本".format(np.count_nonzero(test_gt), np.count_nonzero(gt)))
        else:
            log.info("采样方式：固定样本比例")
            train_gt, test_gt = sample_gt(gt, TRAINING_PERCENTAGE, mode='random', sample_nums=SAMPLE_NUMS)
            log.info("挑选{}个训练样本，总计{}个可用样本".format(np.count_nonzero(train_gt), np.count_nonzero(gt)))
            log.info("挑选{}个测试样本，总计{}个可用样本".format(np.count_nonzero(test_gt), np.count_nonzero(gt)))
        
        mask = np.unique(train_gt)
        tmp = []
        for v in mask:
            tmp.append(np.sum(train_gt==v))
        log.info("数据集类别：{}".format(mask))
        log.info("训练集大小：{}".format(tmp))
        mask = np.unique(test_gt)
        tmp = []
        for v in mask:
            tmp.append(np.sum(test_gt==v))
        log.info("测试集大小：{}".format(tmp))
         
        train_dataset = HyperX(img, train_gt, **hyperparams)
        log.info('训练集数据的形状：{}'.format(train_dataset.data.shape))
        log.info('训练集标签的形状：{}'.format(train_dataset.label.shape))           
        test_dataset = HyperX(img, test_gt, **hyperparams)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                    batch_size=hyperparams['batch_size'],
                                    shuffle=True,
                                    drop_last=False)
        log.info("Train dataloader:{}".format(len(train_loader))) # 9
                
        for k, v in hyperparams.items():
            log.info("{}:{}".format(k, v))
        log.info("Network :")
        with torch.no_grad():
            for input, _ in train_loader:
                break
            summary(model.to(hyperparams['device']), input.size()[1:])
        print(model)

        # 训练阶段
        for epoch in range(args.epochs):
            adjust_learning_rate(optimizer, init_lr, epoch, args) 
            train(train_loader, model, criterion, optimizer, epoch, args, N_CLASSES)
            
        prediction = test(model, img, hyperparams)
        results = metrics(prediction, test_gt, ignored_labels=hyperparams['ignored_labels'], n_classes=hyperparams['n_classes'])
        
        color_gt = display_goundtruth(gt=prediction, vis=None, caption="Testing ground truth(full)" + "RUN{}".format(i+1))
        if args.sample_nums:
            plt.imsave("./result/" + str(datetime.date.today()) + "{} lr{} sample{} patch{} {} finetune {} RUN{} Testing gt(full).png".format(hyperparams['dataset'],hyperparams['lr'],hyperparams['sample_nums'],hyperparams['patch_size'],hyperparams['desc'],hyperparams['fine_tune'],i+1), color_gt)
            mask = np.zeros(gt.shape, dtype='bool')
            for l in IGNORED_LABELS:
                mask[gt == l] = True
            prediction[mask] = 0
            color_gt = display_goundtruth(gt=prediction, vis=None, caption="Testing ground truth(semi)"+"RUN{}".format(i))
            plt.imsave("./result/" + str(datetime.date.today()) + "{} lr{} sample{} patch{} {} finetune {} RUN{} Testing gt(labeled).png".format(hyperparams['dataset'],hyperparams['lr'],hyperparams['sample_nums'],hyperparams['patch_size'],hyperparams['desc'],hyperparams['fine_tune'],i+1), color_gt)
        else:
            plt.imsave("./result/" + str(datetime.date.today()) + "{} lr{} sample{} patch{} {} finetune {} RUN{} Testing gt(full).png".format(hyperparams['dataset'],hyperparams['lr'],TRAINING_PERCENTAGE,hyperparams['patch_size'],hyperparams['desc'],hyperparams['fine_tune'],i+1), color_gt)
            mask = np.zeros(gt.shape, dtype='bool')
            for l in IGNORED_LABELS:
                mask[gt == l] = True
            prediction[mask] = 0
            color_gt = display_goundtruth(gt=prediction, vis=None, caption="Testing ground truth(semi)"+"RUN{}".format(i))
            plt.imsave("./result/" + str(datetime.date.today()) + "{} lr{} sample{} patch{} {} finetune {} RUN{} Testing gt(labeled).png".format(hyperparams['dataset'],hyperparams['lr'],TRAINING_PERCENTAGE,hyperparams['patch_size'],hyperparams['desc'],hyperparams['fine_tune'],i+1), color_gt)
        
        acc_dataset[i,0] = results['Accuracy']
        A[i] = results['F1 scores'][1:]
        K[i,0] = results['Kappa']
        
        log.info('----------Training result----------')
        log.info("\nConfusion matrix:\n{}".format(results['Confusion matrix']))
        log.info("\nAccuracy:\n{:.4f}".format(results['Accuracy']))
        log.info("\nF1 scores:\n{}".format(np.around(results['F1 scores'], 4)))
        log.info("\nKappa:\n{:.4f}".format(results['Kappa']))
        print("Acc_dataset {}".format(acc_dataset))

    
    OA_std = np.std(acc_dataset)
    OAMean = np.mean(acc_dataset)
    AA_std = np.std(A,1)
    AAMean = np.mean(A,1)
    Kappa_std = np.std(K)
    KappaMean = np.mean(K)

    AA = list(map('{:.2f}%'.format, AAMean))
    p = []
    log.info("{}数据集的结果如下".format(DATASET))
    for item,std in zip(AAMean,AA_std):
        p.append(str(round(item*100,2))+"+-"+str(round(std,2)))
    log.info(np.array(p))
    log.info("AAMean {:.2f} +-{:.2f}".format(np.mean(AAMean)*100,np.mean(AA_std)))
    log.info("{}".format(acc_dataset))
    log.info("OAMean {:.2f} +-{:.2f}".format(OAMean,OA_std))
    log.info("{}".format(K))
    log.info("KappaMean {:.2f} +-{:.2f}".format(KappaMean,Kappa_std))
    
   
def train(train_loader, model, criterion, optimizer, epoch, args, N_CLASSES):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    model.eval()
    feature = torch.rand(1,128).cuda() 
    label = torch.rand(1) .cuda()
    end = time.time()
    for i, (images, target ) in enumerate(train_loader):
        data_time.update(time.time() - end)
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        # compute output
        output, flat_emb = model(images)
        feature = torch.cat([feature, flat_emb], dim=0)
        
        label = torch.cat([label, target], dim=0)
        loss = criterion(output, target)
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))  # 32
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i)


def test(net, img, hyperparams):
    """
    Test a model on a specific image
    """
    net.eval()
    patch_size = hyperparams['patch_size']
    center_pixel = hyperparams['center_pixel']
    batch_size, device = hyperparams['batch_size'], hyperparams['device']
    n_classes = hyperparams['n_classes']
    # probs = np.zeros(img.shape[:2] + (n_classes,))
    probs = np.zeros(img.shape[:2])
    img = np.pad(img, ((patch_size // 2, patch_size // 2), (patch_size // 2, patch_size // 2), (0, 0)), 'reflect')
    iterations = count_sliding_window(img, step=hyperparams['test_stride'], window_size=(patch_size, patch_size))
    for batch in tqdm(grouper(batch_size, sliding_window(img, step=1, window_size=(patch_size, patch_size))),
                      total=(iterations//batch_size),
                      desc="Inference on the image"
                      ):
        with torch.no_grad():
            data = [b[0] for b in batch]
            data = np.copy(data)
            data = data.transpose(0, 3, 1, 2)
            data = torch.from_numpy(data)
            # data = data.unsqueeze(1)
            indices = [b[1:] for b in batch]
            data = data.to(device)
            data = data.type(torch.cuda.FloatTensor)
            output, _ = net(data)
            _, output = torch.max(output, dim=1)
            if isinstance(output, tuple):
                output = output[0]
            output = output.to('cpu')
            if center_pixel:
                output = output.numpy()
            else:
                output = np.transpose(output.numpy(), (0, 2, 3, 1))
            for (x, y, w, h), out in zip(indices, output):
                if center_pixel:
                    probs[x, y] += out
                else:
                    probs[x:x + w, y:y + h] += out
    return probs

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

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pred = pred.to(device)
        target = target.to(device)
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()


