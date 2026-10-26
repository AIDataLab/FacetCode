import torch
import torch.nn as nn
import torch.nn.functional as F


def loss_fn(logits, labels, loss_config, avg_label_num=None):
    loss_type = loss_config.get('name', 'sigce')
    if loss_type == 'ce':
        return F.cross_entropy(logits, labels)
    elif loss_type == 'focal':
        gamma = loss_config.get('gamma', 1.0)
        alpha = loss_config.get('alpha', 1.0)
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction='none')
        p_t = probs * labels.float() + (1 - probs) * (1 - labels.float())
        focal_weight = (1 - p_t) ** gamma
        loss = focal_weight * bce
        if alpha is not None:
            alpha_t = alpha * labels.float() + (1 - alpha) * (1 - labels.float())
            loss = alpha_t * loss
        return loss.mean()
    elif loss_type == 'sigce' or loss_type == 'sigfocal':
        if avg_label_num is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction='none')
            loss = (loss.sum(dim=1) / avg_label_num).mean()
        else:
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        return loss
    elif loss_type == 'sigfocal':
        gamma = loss_config.get('gamma', 1.0)
        alpha = loss_config.get('alpha', 1.0)
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction='none')
        p_t = probs * labels.float() + (1 - probs) * (1 - labels.float())
        focal_weight = (1 - p_t) ** gamma
        loss = focal_weight * bce
        if alpha is not None:
            alpha_t = alpha * labels.float() + (1 - alpha) * (1 - labels.float())
            loss = alpha_t * loss
        if avg_label_num is not None:
            loss = (loss.sum(dim=1) / avg_label_num).mean()
        else:
            loss = loss.mean()
        return loss
    elif loss_type == 'asy':
        gamma_neg = loss_config.get('gamma_neg', 4.0)
        gamma_pos = loss_config.get('gamma_pos', 1.0)
        clip = loss_config.get('clip', 0.05)
        disable_torch_grad_focal_loss = loss_config.get('disable_torch_grad_focal_loss', False)

        probs = torch.sigmoid(logits)
        targets = labels.float()

        gamma_neg_loss = ((1 - targets) * ((1 - probs) ** gamma_neg) * (-torch.log(probs + 1e-8)))
        gamma_pos_loss = (targets * ((probs ** gamma_pos) * (-torch.log(1 - probs + 1e-8))))

        asym_loss = gamma_neg_loss + gamma_pos_loss

        if clip is not None and clip > 0:
            asym_loss = torch.clamp(asym_loss, min=clip, max=None)

        if avg_label_num is not None:
            loss = (asym_loss.sum(dim=1) / avg_label_num).mean()
        else:
            loss = asym_loss.mean()
        return loss
    else:
        if avg_label_num is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction='none')
            loss = (loss.sum(dim=1) / avg_label_num).mean()
        else:
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        return loss