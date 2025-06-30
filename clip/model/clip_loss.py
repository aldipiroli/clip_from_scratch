import torch.nn as nn


class CLIPLoss(nn.Module):
    def __init__(self):
        super(self).__init__()

    def forward(self, pred, gt):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, gt)
        return loss
