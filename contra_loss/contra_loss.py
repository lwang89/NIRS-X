import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCLR(nn.modules.loss._Loss):
    """
    replicate from the original paper, tested.
    """

    def __init__(self, device="cuda", T=1.0):
        super().__init__()
        self.device = device
        self.T = T

    def forward(self, embedding_anchor, embedding_positive):
        """
        :param embedding_anchor:
        :param embedding_positive:
        :return: a tensor, we can use loss.item() to get the value
        """

        # L2 normalization
        norm_embedding_anchor = F.normalize(embedding_anchor)
        norm_embedding_positive = F.normalize(embedding_positive)

        batch_size = embedding_anchor.shape[0]
        embedding_total = torch.cat([norm_embedding_anchor, norm_embedding_positive], dim=0)

        # representation similarity matrix, 2Nx2N
        logits = torch.mm(embedding_total, embedding_total.t()).to(self.device)
        logits.fill_diagonal_(-1e10)
        logits /= self.T

        # find all positive pairs
        targets = torch.LongTensor(torch.cat([torch.arange(batch_size, 2 * batch_size), torch.arange(batch_size)])).to(self.device)
        loss = F.cross_entropy(logits, targets)
        return loss


class SimSiam(nn.modules.loss._Loss):
    """
    replicate from the original paper
    """

    def __init__(self, device="cuda", T=0.07):
        super().__init__()
        self.device = device
        self.T = T
        self.criterion = nn.CosineSimilarity(dim=1).cuda("gpu")

    def forward(self, p1, p2, z1, z2):
        loss = -(self.criterion(p1, z2).sum() + self.criterion(p2, z1).sum()) * 0.5

        return loss
