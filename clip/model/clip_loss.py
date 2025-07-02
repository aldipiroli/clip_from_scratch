import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits_img, logits_text):
        B, C = logits_img.shape
        labels = torch.arange(B).to(logits_img.device)

        loss_img2text = F.cross_entropy(logits_img, labels)
        loss_text2img = F.cross_entropy(logits_text, labels)
        loss = (loss_img2text + loss_text2img) / 2
        return loss
