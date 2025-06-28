
import math
import torch
import torch.nn as nn
import torch.distributed as dist
from training.train_utils import AverageMeter, ProgressMeter, accuracy
import copy
import time

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
    model.train()
    
    if epoch<args.warmup_epochs:
        cur_memory_lr =  args.memory_lr* (epoch+1) / args.warmup_epochs 
    elif args.memory_lr != args.memory_lr_final:
        cur_memory_lr = args.memory_lr_final + 0.5 * \
                   (1. + math.cos(math.pi * (epoch-args.warmup_epochs) / (args.epochs-args.warmup_epochs))) \
                   * (args.memory_lr- args.memory_lr_final)
    else:
        cur_memory_lr = args.memory_lr
    cur_adco_t =args.mem_t
    end = time.time()

    for i, (data0, _, data1, _) in enumerate(train_loader):
        images = [data0, data1]
        data_time.update(time.time() - end)
        if args.gpu is not None:
            for k in range(len(images)):
                images[k] = images[k].cuda(args.gpu, non_blocking=True)
        
        update_sym_network(model, images, args, Memory_Bank, Memory_Bank_upper, losses, top1, top5,
                           optimizer, criterion, mem_losses,moco_momentum,cur_memory_lr,cur_adco_t)
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0 and args.rank==0:
            progress.display(i)
            if args.rank == 0:
                progress.write(train_log_path, i)    
    return top1.avg           


# update online encoder
def update_sym_network(model, images, args, Memory_Bank, Memory_Bank_upper,
                   losses, top1, top5, optimizer, criterion, mem_losses,
                   moco_momentum,memory_lr,cur_adco_t):
    model.zero_grad()

    # 四个对象均是torch.Size([batch_size, moco_dim])，在CaCo.py的forward_withoutpred_sym方法处返回
    # q_pred和k_pred是image1和image2经过online encoder的结果，q和k是image1和image2经过key encoder的结果
    q_pred, k_pred, q, k, q_pred_upper, k_pred_upper, q_upper, k_upper = model(im_q=images[0], im_k=images[1],run_type=0,moco_momentum=moco_momentum)
    
    d_norm1_upper, d1_upper, logits1_upper = Memory_Bank_upper(q_pred_upper)
    d_norm2_upper, d2_upper, logits2_upper = Memory_Bank_upper(k_pred_upper)
    with torch.no_grad():
        logits1_upper_keep = logits1_upper.clone()
        logits2_upper_keep = logits2_upper.clone()
    logits1_upper /= args.moco_upper_t 
    logits2_upper /= args.moco_upper_t

    # 浅层字典检索正原型
    with torch.no_grad():
        _, _, check_logits1_upper = Memory_Bank_upper(k_upper)
        logits_fix1_upper = copy.deepcopy(check_logits1_upper)
        check_logits1_upper = check_logits1_upper.detach()
        filter_index1_upper = torch.argmax(check_logits1_upper, dim=1)
        labels1_upper = copy.deepcopy(filter_index1_upper)
        filter_findex1_upper = torch.argmin(check_logits1_upper, dim=1)
        flabels1_upper = copy.deepcopy(filter_findex1_upper)
        _, _, check_logits2_upper = Memory_Bank_upper(q_upper)
        logits_fix2_upper = copy.deepcopy(check_logits2_upper)
        check_logits2_upper = check_logits2_upper.detach()
        filter_index2_upper = torch.argmax(check_logits2_upper, dim=1)
    
    d_norm1, d1, logits1 = Memory_Bank(q_pred)
    d_norm2, d2, logits2 = Memory_Bank(k_pred)

    # logits: Nx(1+K)
    with torch.no_grad():
        logits_keep1 = logits1.clone()
        logits_keep2 = logits2.clone()
    
    logits1 /= args.moco_t 
    logits2 /= args.moco_t 

    # 终端字典检索正原型
    with torch.no_grad():
        d_norm21, d21, check_logits1 = Memory_Bank(k)
        logits_fix1 = copy.deepcopy(check_logits1)
        check_logits1 = check_logits1.detach()
        filter_index1 = torch.argmax(check_logits1, dim=1)
        labels1 = copy.deepcopy(filter_index1)
        filter_findex1 = torch.argmin(check_logits1, dim=1)
        flabels1 = copy.deepcopy(filter_findex1)
        check_logits1 = logits_fix1
        d_norm22, d22, check_logits2 = Memory_Bank(q)
        check_logits2 = check_logits2.detach()
        logits_fix2 = check_logits2
        filter_index2 = torch.argmax(check_logits2, dim=1)
    
    # 损失函数
    CaCo_loss = criterion(logits1, labels1)
    HiCo_loss = criterion(logits1, labels1) + criterion(logits1_upper, labels1_upper) 
    HiBiCo_loss = criterion(logits1, labels1) + criterion(logits1_upper, labels1_upper) - criterion(logits1, flabels1) * 0.1 - criterion(logits1_upper, flabels1_upper) * 0.1

    if args.loss=="CaCo":
        loss = CaCo_loss
    elif args.loss=='HiCo':
        loss = HiCo_loss
    elif args.loss=='HiBiCo':
        loss = HiBiCo_loss
    else:
        print("请设置loss超参!")
    
    acc1, acc5 = accuracy(logits1, labels1, topk=(1, 5))
    losses.update(loss.item(), images[0].size(0))
    top1.update(acc1.item(), images[0].size(0))
    top5.update(acc5.item(), images[0].size(0))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # update shallow dictionary
    with torch.no_grad():
        logits1_upper = logits1_upper_keep/cur_adco_t
        logits2_upper = logits2_upper_keep/cur_adco_t
        p_qd1_upper = nn.functional.softmax(logits1_upper, dim=1)
        p_qd1_upper[torch.arange(logits1_upper.shape[0]), filter_index1_upper] = 1 - p_qd1_upper[torch.arange(logits1_upper.shape[0]), filter_index1_upper]
        g1_upper = torch.einsum('cn,nk->ck', [q_pred_upper.T, p_qd1_upper]) / logits1_upper.shape[0] - torch.mul(
            torch.mean(torch.mul(p_qd1_upper, logits1_upper_keep), dim=0), d_norm1_upper)
        p_qd2_upper = nn.functional.softmax(logits2_upper, dim=1)
        p_qd2_upper[torch.arange(logits1_upper.shape[0]), filter_index2_upper] = 1 - p_qd2_upper[torch.arange(logits2_upper.shape[0]), filter_index2_upper]
        g2_upper = torch.einsum('cn,nk->ck', [k_pred_upper.T, p_qd2_upper]) / logits2_upper.shape[0] - torch.mul(
            torch.mean(torch.mul(p_qd2_upper, logits2_upper_keep), dim=0), d_norm2_upper)
        g_upper = -torch.div(g1_upper, torch.norm(d1_upper, dim=0))  - torch.div(g2_upper,torch.norm(d2_upper, dim=0))
        g_upper /=cur_adco_t
        Memory_Bank_upper.v.data = args.mem_momentum * Memory_Bank_upper.v.data + g_upper 
        Memory_Bank_upper.W.data = Memory_Bank_upper.W.data - memory_lr * Memory_Bank_upper.v.data
    
    # update terminal dictionary
    with torch.no_grad():
        logits1 = logits_keep1/cur_adco_t
        logits2 = logits_keep2/cur_adco_t
        p_qd1 = nn.functional.softmax(logits1, dim=1)
        p_qd1[torch.arange(logits1.shape[0]), filter_index1] = 1 - p_qd1[torch.arange(logits1.shape[0]), filter_index1]
        g1 = torch.einsum('cn,nk->ck', [q_pred.T, p_qd1]) / logits1.shape[0] - torch.mul(
            torch.mean(torch.mul(p_qd1, logits_keep1), dim=0), d_norm1)
        p_qd2 = nn.functional.softmax(logits2, dim=1)
        p_qd2[torch.arange(logits1.shape[0]), filter_index2] = 1 - p_qd2[torch.arange(logits2.shape[0]), filter_index2]
        g2 = torch.einsum('cn,nk->ck', [k_pred.T, p_qd2]) / logits2.shape[0] - torch.mul(
            torch.mean(torch.mul(p_qd2, logits_keep2), dim=0), d_norm2)
        g = -torch.div(g1, torch.norm(d1, dim=0))  - torch.div(g2,torch.norm(d2, dim=0))#/ args.mem_t  # c*k
        g /=cur_adco_t
        Memory_Bank.v.data = args.mem_momentum * Memory_Bank.v.data + g #+ args.mem_wd * Memory_Bank.W.data
        Memory_Bank.W.data = Memory_Bank.W.data - memory_lr * Memory_Bank.v.data
    
    with torch.no_grad():
        logits = torch.softmax(logits1, dim=1)
        posi_prob = logits[torch.arange(logits.shape[0]), filter_index1]
        posi_prob = torch.mean(posi_prob)
        mem_losses.update(posi_prob.item(), logits.size(0))
    

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
