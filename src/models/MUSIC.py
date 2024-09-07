from torch import nn
import torch
import math


def NegCELoss(neg_logits, neg_labels):
    soft = nn.Softmax(1)
    q1 = 1 - soft(neg_logits)
    q = soft(q1)
    weight = 1 - q
    out = weight * math.log(q)
    return nn.NLLLoss(out, neg_labels)


def mini_entropy_loss(p):
    soft = nn.Softmax(1)
    p = soft(p)
    epsilon = 1e-5  # avoid log(0)
    return -1 * torch.sum(p * math.log(p + epsilon)) / p.shape(0)
