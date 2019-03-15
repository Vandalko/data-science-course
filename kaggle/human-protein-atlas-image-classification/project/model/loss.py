import torch.nn.functional as F


def focal_loss(logit, target):
    gamma = 2
    target = target.float()
    max_val = (-logit).clamp(min=0)
    loss = logit - logit * target + max_val + ((-max_val).exp() + (-logit - max_val).exp()).log()

    invprobs = F.logsigmoid(-logit * (target * 2.0 - 1.0))
    loss = (invprobs * gamma).exp() * loss
    if len(loss.size()) == 2:
        loss = loss.sum(dim=1)
    return loss.mean()
