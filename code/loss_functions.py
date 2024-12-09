import torch
import torch.nn.functional as F
import numpy as np

import constants as c

def combined_loss(gen_frames, gt_frames, d_preds, lam_adv=1, lam_lp=1, lam_gdl=1, l_num=2, alpha=2, device='cuda'):
    batch_size = gen_frames[0].shape[0]

    loss = lam_lp * lp_loss(gen_frames, gt_frames, l_num)
    loss += lam_gdl * gdl_loss(gen_frames, gt_frames, alpha)
    if c.ADVERSARIAL: loss += lam_adv * adv_loss(d_preds, torch.ones([batch_size, 1]).to(device))


    return loss

def bce_loss(preds, targets):
    return F.binary_cross_entropy(preds, targets, reduction='sum')

def lp_loss(gen_frames, gt_frames, l_num):
    scale_losses = []
    for i in range(len(gen_frames)):
        scale_losses.append(torch.sum(torch.abs(gen_frames[i] - gt_frames[i])**l_num))

    return torch.mean(torch.stack(scale_losses))

def gdl_loss(gen_frames, gt_frames, alpha):
    """
    Compute the sum of GDL losses between the generated frames and the ground truth frames.
    :param gen_frames: (batch_size, channels, height, width)
    :param gt_frames: (batch_size, channels, height, width)
    :param alpha:
    :return:
    """
    scale_losses = []
    for i in range(len(gen_frames)):
        gen_dx = torch.abs(torch.gradient(gen_frames[i], dim=3)[0])
        gen_dy = torch.abs(torch.gradient(gen_frames[i], dim=2)[0])
        gt_dx = torch.abs(torch.gradient(gt_frames[i], dim=3)[0])
        gt_dy = torch.abs(torch.gradient(gt_frames[i], dim=2)[0])

        grad_diff_x = torch.abs(gt_dx - gen_dx)
        grad_diff_y = torch.abs(gt_dy - gen_dy)

        loss = torch.sum(torch.pow(grad_diff_y, alpha) + torch.pow(grad_diff_x, alpha))
        scale_losses.append(loss)

    return torch.mean(torch.Tensor(scale_losses))

def adv_loss(preds, labels):
    scale_losses = []
    for i in range(len(preds)):
        loss = bce_loss(preds[i], labels)
        scale_losses.append(loss)

    return torch.mean(torch.stack(scale_losses))