import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def CE_Loss(inputs, target, cls_weights, num_classes=21):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    CE_loss = nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes)(temp_inputs, temp_target)
    return CE_loss


def Focal_Loss(inputs, target, cls_weights, num_classes=21, alpha=0.5, gamma=2):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    logpt = -nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes, reduction='none')(temp_inputs,
                                                                                                 temp_target)
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    loss = loss.mean()
    return loss


def Dice_loss(inputs, target, beta=1, smooth=1e-5):
    """
    计算Dice Loss

    Args:
        inputs: 预测结果，shape为 [n, c, h, w]
        target: 真实标签，shape为 [n, h, w] (类别索引) 或 [n, h, w, c] (one-hot)
        beta: Dice系数参数，beta=1为标准Dice，beta>1更关注recall
        smooth: 平滑项，防止除零
    """
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()

    # 如果尺寸不匹配，进行插值
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    # 如果target是类别索引格式，转换为one-hot格式
    if len(target.shape) == 3:
        # target shape: [n, h, w]
        target_one_hot = torch.zeros(n, c, ht, wt, device=target.device)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1)
    else:
        # target已经是one-hot格式
        target_one_hot = target.permute(0, 3, 1, 2) if target.shape[-1] == c else target

    # 对inputs应用softmax
    inputs_softmax = torch.softmax(inputs, dim=1)

    # 计算每个类别的Dice系数
    intersection = torch.sum(inputs_softmax * target_one_hot, dim=(2, 3))
    union = torch.sum(inputs_softmax + target_one_hot, dim=(2, 3))

    # Dice系数
    dice = (2. * intersection + smooth) / (union + smooth)

    # Dice Loss = 1 - mean(Dice)
    dice_loss = 1 - dice.mean()

    return dice_loss


def CombinedLoss(inputs, target, cls_weights, num_classes=21, dice_weight=0.3, ce_weight=1.0):
    """
    组合CE Loss和Dice Loss

    Args:
        inputs: 预测结果
        target: 真实标签
        cls_weights: 类别权重
        num_classes: 类别数量
        dice_weight: Dice Loss权重
        ce_weight: CE Loss权重
    """
    # 计算CE Loss
    ce_loss = CE_Loss(inputs, target, cls_weights, num_classes)

    # 计算Dice Loss
    dice_loss = Dice_loss(inputs, target)

    # 组合损失
    total_loss = ce_weight * ce_loss + dice_weight * dice_loss

    return total_loss, ce_loss, dice_loss


def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s type' % init_type)
    net.apply(init_func)


def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio=0.1, warmup_lr_ratio=0.1,
                     no_aug_iter_ratio=0.3, step_num=10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                    1.0 + math.cos(
                math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n = iters // step_size
        out_lr = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr