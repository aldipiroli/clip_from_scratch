import torch
import torch.nn as nn
from torchvision.models import vit_b_16


class TextEncoder(nn.Module):
    def __init__(self, context_len, embed_size, vocab_size):
        super().__init__()
        self.context_len = context_len
        self.embed_size = embed_size
        self.vocab_size = vocab_size

    def forward(self, x):
        return x


class ViTb16FeatureExtractor(torch.nn.Module):
    def __init__(self, pretrained=True, use_cls_token=False):
        super().__init__()
        # https://docs.pytorch.org/vision/main/_modules/torchvision/models/vision_transformer.html#vit_b_16
        # Note: patch_size: 16, img_size: (224,224)
        self.vit = vit_b_16(weights="DEFAULT" if pretrained else None)
        self.use_cls_token = use_cls_token

    def forward(self, x):
        x = self.vit._process_input(x)
        n = x.shape[0]
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = x + self.vit.encoder.pos_embedding
        x = self.vit.encoder.dropout(x)
        for layer in self.vit.encoder.layers:
            x = layer(x)
        if not self.use_cls_token:
            x = x[:, 1:, :]
        return x  # (B, C, H, W) ->  (B, img_size//patch_size, patch_size**2*3) = (B, 197, 768)


class ImgEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.encoder = ViTb16FeatureExtractor(pretrained)

    def forward(self, x):
        out = self.encoder(x)
        return out


class CLIPModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, x):
        return x
