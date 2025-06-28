import warnings
import torch
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
    model.train()
    for i, (data0, _, data1, _) in enumerate(train_loader):
        images = [data0, data1]
        if args.dataset=='IndianPines' or args.dataset=='HyRANK' or args.dataset=='Botswana' or args.dataset=='Houston18':
            if args.gpu is not None:
                for k in range(len(images)):  
                    images[k] = images[k].cuda(args.gpu, non_blocking=True)
        q, _, _, k, q_upper, _, _, k_upper  = model(im_q=images[0], im_k=images[1])
        q_upper = q_upper.view(q_upper.size(0), -1)
        k_upper = k_upper.view(k_upper.size(0), -1)

        _, _, l_neg_upper = Memory_Bank_upper(q_upper)
        l_pos_upper = torch.einsum('nc,nc->n', [q_upper, k_upper]).unsqueeze(-1)
        logits_upper = torch.cat([l_pos_upper, l_neg_upper], dim=1)
        logits_upper /= 0.2
        labels_upper = torch.zeros(logits_upper.shape[0], dtype=torch.long).cuda(args.gpu)

        d_norm, d, l_neg = Memory_Bank(q)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= 0.2  # using the default param in MoCo temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda(args.gpu)
        loss = criterion(logits, labels) + criterion(logits_upper, labels_upper)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1.item(), images[0].size(0))
        top5.update(acc5.item(), images[0].size(0))
        if i % args.print_freq == 0 and args.rank==0:
            progress.display(i)
        output = k
        batch_size = output.size(0)
        start_point = i * batch_size
        end_point = min((i + 1) * batch_size, args.cluster)
        Memory_Bank.W.data[:, start_point:end_point] = output[:end_point - start_point].T
        
        # update shallow memory bank
        output_upper = k_upper
        batch_size_upper = output_upper.size(0)
        start_point_upper = i * batch_size_upper
        end_point_upper = min((i + 1) * batch_size_upper, args.cluster_upper)
        Memory_Bank_upper.W.data[:, start_point_upper:end_point_upper] = k_upper[:end_point_upper - start_point_upper].T
        if (i+1) * batch_size >= args.cluster:
            break
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