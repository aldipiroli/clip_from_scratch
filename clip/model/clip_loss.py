import torch.nn as nn
import torch.nn.functional as F


class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits_img, logits_text, labels):
        B, C = logits_img.shape

        loss_img2text = F.cross_entropy(logits_img, labels)
        loss_text2img = F.cross_entropy(logits_text, labels)
        loss = (loss_img2text + loss_text2img) / 2
        loss_dict = {
            "loss/img2text": loss_img2text.item(),
            "loss/text2img": loss_text2img.item(),
            "loss/total": loss.item(),
        }
        return loss, loss_dict
