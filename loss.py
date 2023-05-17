import torch
import torch.nn.functional as F


def inv_dec_loss(h1, h2, lambd):
    N = h1.size(0)

    c = torch.mm(h1.T, h2)
    c1 = torch.mm(h1.T, h1)
    c2 = torch.mm(h2.T, h2)

    c = c / N
    c1 = c1 / N
    c2 = c2 / N

    loss_inv = -torch.diagonal(c).sum()
    iden = torch.eye(c.shape[0]).to(h1.device)
    loss_dec1 = (iden - c1).pow(2).sum()
    loss_dec2 = (iden - c2).pow(2).sum()

    loss = loss_inv + lambd * (loss_dec1 + loss_dec2)

    return loss