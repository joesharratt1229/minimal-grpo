import torch
import torch.nn as nn

def approximate_kl_divergence_loss(probs, ref_probs, action_mask):
    log_ratio = (probs - ref_probs)
    return log_ratio.exp() - (log_ratio) - 1

def masked_mean(tensor, mask, dim=None):
    if mask is None:
        return tensor.mean(axis=dim)
    return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim)


class GRPOLoss(nn.Module):
    def __init__(self, clip_ratio, kl_weight):
        self.eps = 1e-8
        self.clip_ratio = clip_ratio
        self.kl_weight = kl_weight

    def forward(self, log_probs, old_probs, log_probs_ref, advantanges, action_mask):
        likelihood_ratio = (log_probs - old_probs).exp()
        clipped_obj = torch.clamp(likelihood_ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantanges
        kl_loss = approximate_kl_divergence_loss(log_probs, log_probs_ref, action_mask)
        obj = likelihood_ratio * advantanges
        loss = -torch.min(obj, clipped_obj)
        loss = masked_mean(loss, action_mask, dim = -1).mean()
        loss += self.kl_weight * kl_loss
        return loss, kl_loss
        
        
        
        
        