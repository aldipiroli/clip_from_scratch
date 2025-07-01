import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16


class AttentionLayer(nn.Module):
    def __init__(self, embed_size, embed_size_h, dropout):
        super().__init__()
        self.embed_size = embed_size
        self.embed_size_h = embed_size_h

        self.Q = nn.Linear(embed_size, embed_size_h)
        self.K = nn.Linear(embed_size, embed_size_h)
        self.V = nn.Linear(embed_size, embed_size_h)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.K(x)  # (B, T, C)
        q = self.Q(x)
        v = self.V(x)

        qk = q @ k.transpose(2, 1) * self.embed_size_h**-0.5
        tril = torch.tril(torch.ones(T, T))
        qk = qk * tril
        qk = qk.masked_fill(qk == 0, float("-inf"))
        qk_smax = F.softmax(qk, -1)
        assert (qk_smax == qk_smax).all()  # nan sanity check
        qk = self.dropout(qk)
        out = qk_smax @ v
        return out


class TransformerLayer(nn.Module):
    def __init__(self, embed_size, n_heads, dropout):
        super().__init__()
        assert embed_size % n_heads == 0

    def forward(self, x):
        return x


class TextEncoder(nn.Module):
    def __init__(self, context_len, embed_size, vocab_size):
        super().__init__()
        self.context_len = context_len
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.text_embeddings = nn.Embedding(vocab_size, embed_size)
        self.pos_embeddings = nn.Embedding(context_len, embed_size)

    def forward(self, x):
        x_embed = self.text_embeddings(x)
        x_pos_embed = self.pos_embeddings(x)
        x = x_embed + x_pos_embed

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
        self.img_encoder = ImgEncoder(config["img_encoder"]["pretrained"])

    def forward(self, x):
        return x
