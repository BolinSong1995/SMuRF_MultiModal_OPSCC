import pandas as pd
import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from lifelines.utils import concordance_index
from typing import Tuple
from math import ceil

def extract_csv(file):
    '''From csv file path, returns the features and labels '''
    df = pd.read_csv(file)
    return df

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def define_optimizer(args, model):
    optimizer = None
    if args.optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(
            args.beta1, args.beta2), weight_decay=args.weight_decay)
    elif args.optimizer_type == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(
        ), lr=args.lr, weight_decay=args.weight_decay, initial_accumulator_value=0.1)
    else:
        raise NotImplementedError(
            'initialization method [%s] is not implemented' % args.optimizer)
    return optimizer


def define_scheduler(args, optimizer):
    if args.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + args.epoch_count -
                             args.niter) / float(100 + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'exp':
        scheduler = lr_scheduler.ExponentialLR(optimizer, 0.1, last_epoch=-1)
    elif args.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=args.lr_decay_iters, gamma=0.1)
    elif args.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif args.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.niter, eta_min=0)
    elif args.lr_policy == 'constant':
        scheduler = lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=1)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


def define_act_layer(act_type='relu'):
    if act_type == 'tanh':
        act_layer = nn.Tanh()
    elif act_type == 'relu':
        act_layer = nn.ReLU()
    elif act_type == 'gelu':
        act_layer = nn.GELU()
    elif act_type == 'sigmoid':
        act_layer = nn.Sigmoid()
    elif act_type == 'LSM':
        act_layer = nn.LogSoftmax(dim=1)
    elif act_type == "none":
        act_layer = None
    else:
        raise NotImplementedError(
            'activation layer [%s] is not found' % act_type)
    return act_layer


def compute_metrics(args, preds):
    preds_grade, preds_hazard, y, time, event, ID = preds
    if args.task=="multitask":
        preds_grade = preds_grade.cpu().detach().numpy()
        # print(y)
        # print(time)
        # print(event)
        y = y.cpu().detach().numpy()
        preds_hazard = preds_hazard.cpu().detach().numpy()
        time = time.cpu().detach().numpy()
        event = event.cpu().detach().numpy()
        ci = concordance_index(time, -preds_hazard, event)
        auc = roc_auc_score(y, preds_grade)
        return ci, auc
    elif args.task=="grade":
        preds_grade = preds_grade.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        auc = roc_auc_score(y, preds_grade)
        return 0, auc
    elif args.task == "survival":
        preds_hazard = preds_hazard.cpu().detach().numpy()
        time = time.cpu().detach().numpy()
        event = event.cpu().detach().numpy()
        ci = concordance_index(time, -preds_hazard, event)
        return ci, 0
    else:
        raise NotImplementedError(
            f'task method {args.task} is not implemented')
    
    


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


CT_WINDOWS = {
    "bone": (1800, 400),
    "lung": (1500, -600),
    "soft_tissue": (400, 50),
    "default": (2048, 0)
}


def _w_to_t(ww: int, wl: int) -> Tuple[float, float]:
    """Convert Window width / Window level.
    Parameters
    ----------
    ww : int
        Window width
    wl : int
        Window level
    Returns
    -------
    Tuple[int,int]
        Lower and upper threshold to use for clipping array values
    """
    upper = wl + (ww / 2)
    lower = wl - (ww / 2)
    return lower, upper


def adjust_ct_window(image, ww, wl):
    """Perform windows adjustement like a radiologist do to visualize its image.

    We also perform quantization, to be more robust to differences in images due to the scanning machine.
    Concretely, once rescaled between 0 to 255, the values are converted to int8 (effectively removing all decimal values)
    then converted back to float32 (for further processing by the model)
    """
    window_min, window_max = _w_to_t(ww, wl)
    if isinstance(image, np.ndarray):
        windowed_img = np.clip(image, window_min, window_max)
    else:
        raise
    windowed_img = (windowed_img - window_min) / (window_max - window_min)
    return windowed_img.astype(np.float32)


def lung_window(image):
    return adjust_ct_window(image, *CT_WINDOWS["lung"])


def bone_window(image):
    return adjust_ct_window(image, *CT_WINDOWS["bone"])


def soft_tissue_window(image):
    return adjust_ct_window(image, *CT_WINDOWS["soft_tissue"])


def default_window(image):
    return adjust_ct_window(image, *CT_WINDOWS["default"])

def center_crop(img, dim):
    
    h, w, d = img.shape[0], img.shape[1], img.shape[2]
    crop_height = dim[1] if dim[1] < img.shape[0] else img.shape[0]
    crop_width = dim[0] if dim[0] < img.shape[1] else img.shape[1]
    crop_depth = dim[2] if dim[2] < img.shape[2] else img.shape[2]
    
    mid_x, mid_y, mid_z = int(w/2), int(h/2), int(d/2)
    cw2, ch2, cd2 = int(crop_width/2), int(crop_height/2), int(crop_depth/2)
    crop_img = img[mid_y-ch2:mid_y+ch2, mid_x -
                   cw2:mid_x+cw2, mid_z-cd2:mid_z+cd2]
    
    return crop_img

def random_crop(img, dim, center):

    crop_height = dim[1] if dim[1] < img.shape[0] else img.shape[0]
    crop_width = dim[0] if dim[0] < img.shape[1] else img.shape[1]
    crop_depth = dim[2] if dim[2] < img.shape[2] else img.shape[2]

    mid_x, mid_y, mid_z = center[1], center[0], center[2]
    cw2, ch2, cd2 = int(crop_width/2), int(crop_height/2), ceil(crop_depth/2)
    crop_img = img[mid_y-ch2:mid_y+ch2, mid_x -
                   cw2:mid_x+cw2, mid_z-cd2:mid_z+cd2]

    return crop_img
