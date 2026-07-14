import torch
import torch.nn as nn
import random
import numpy as np
import os

class BCEDiceLoss(nn.Module):
    def __init__(self, dice_weight=1.5): 
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        smooth = 1e-5
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice_loss = 1.0 - ((2. * intersection + smooth) / (union + smooth)).mean()
        return bce_loss + self.dice_weight * dice_loss

class JointRoutingLoss(nn.Module):

    def __init__(self, dice_weight=1.5, lambda_cls=0.1):
        super(JointRoutingLoss, self).__init__()
        self.seg_criterion = BCEDiceLoss(dice_weight)
        self.cls_criterion = nn.CrossEntropyLoss()
        self.lambda_cls = lambda_cls

    def forward(self, mask_logits, mask_targets, task_logits, task_targets):
        l_seg = self.seg_criterion(mask_logits, mask_targets)
        l_cls = self.cls_criterion(task_logits, task_targets)
        l_total = l_seg + self.lambda_cls * l_cls
        return l_total, l_seg, l_cls

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)