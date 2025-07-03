import torch
import torch.nn as nn
import torch.nn.functional as F
from model.tokenizer import Tokenizer
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
        tril = torch.tril(torch.ones(T, T)).to(x.device)
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
        self.embed_size = embed_size
        self.mha = nn.ModuleList(
            [
                AttentionLayer(embed_size=embed_size, embed_size_h=embed_size // n_heads, dropout=dropout)
                for _ in range(n_heads)
            ]
        )
        self.project = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size), nn.ReLU(), nn.Linear(4 * embed_size, embed_size), nn.Dropout(dropout)
        )

        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        x_mha = torch.cat([h(self.ln1(x)) for h in self.mha], -1)
        x = x_mha + x

        x_project = self.project(self.ln2(x))
        x = x_project + x
        return x


class TextEncoder(nn.Module):
    def __init__(self, context_len, embed_size, vocab_size, n_heads, n_layers, dropout):
        super().__init__()
        self.context_len = context_len
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.text_embeddings = nn.Embedding(vocab_size, embed_size)
        self.pos_embeddings = nn.Embedding(context_len, embed_size)
        self.layers = nn.ModuleList(
            [TransformerLayer(embed_size=embed_size, n_heads=n_heads, dropout=dropout) for _ in range(n_layers)]
        )
        self.ln = nn.LayerNorm(embed_size)
        self.project = nn.Linear(embed_size, vocab_size)

    def forward(self, x, eos_index=None):
        B, T = x.shape
        x_embed = self.text_embeddings(x)
        x_pos_embed = self.pos_embeddings(torch.arange(T, device=x.device))
        x = x_embed + x_pos_embed

        for layer in self.layers:
            x = layer(x)
        x = self.ln(x)  # N, T, embed_size

        if eos_index is not None:
            eos_index = eos_index.reshape(B, 1)
            x = x[torch.arange(B).unsqueeze(1), eos_index, :].squeeze(1)  # N, vocab_size
        return x


class ViTb16FeatureExtractor(torch.nn.Module):
    def __init__(self, pretrained=True, use_cls_token=False, frozen_backbone=True):
        super().__init__()
        # https://docs.pytorch.org/vision/main/_modules/torchvision/models/vision_transformer.html#vit_b_16
        # Note: patch_size: 16, img_size: (224,224)
        self.vit = vit_b_16(weights="DEFAULT" if pretrained else None)
        self.use_cls_token = use_cls_token
        self.frozen_backbone = frozen_backbone
        if frozen_backbone:
            self.vit.eval()

    def forward(self, x):
        with torch.no_grad() if self.frozen_backbone else torch.enable_grad():
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
    def __init__(self, pretrained=True, frozen_backbone=True):
        super().__init__()
        self.encoder = ViTb16FeatureExtractor(pretrained, use_cls_token=True, frozen_backbone=frozen_backbone)
        self.patch_size = 16

    def forward(self, x):
        out = self.encoder(x)
        out = out[:, 0, :]  # N, embed_size
        return out


class CLIPModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        cfg = config["MODEL"]
        cfg_img_enc = cfg["img_encoder"]
        cfg_text_enc = cfg["text_encoder"]

        self.img_encoder = ImgEncoder(
            pretrained=cfg_img_enc["pretrained"], frozen_backbone=cfg_img_enc["frozen_backbone"]
        )
        self.text_encoder = TextEncoder(
            context_len=cfg_text_enc["context_len"],
            embed_size=cfg_text_enc["embed_size"],
            n_heads=cfg_text_enc["n_heads"],
            n_layers=cfg_text_enc["n_layers"],
            dropout=cfg_text_enc["dropout"],
            vocab_size=Tokenizer().get_vocab_size(),
        )
        self.project_img_enc = nn.Linear(self.img_encoder.patch_size**2 * 3, cfg["common_embed_size"])
        self.project_text_enc = nn.Linear(cfg_text_enc["embed_size"], cfg["common_embed_size"])
        self.ln1 = nn.LayerNorm(cfg["common_embed_size"])
        self.ln2 = nn.LayerNorm(cfg["common_embed_size"])
        self.t = nn.Parameter(torch.tensor([torch.log(torch.tensor(1 / 0.07))], dtype=torch.float32))

    def forward(self, img, text, eos_id):
        B, T = text.shape
        img_enc = self.img_encoder(img)  # (B, C, H, W) -> (B, e_img)
        text_enc = self.text_encoder(text, eos_id)  # (B, T) -> (B, e_text)

        img_enc_common = self.ln1(self.project_img_enc(img_enc))  # (B, e_img) -> (B, e_comm)
        text_enc_common = self.ln2(self.project_text_enc(text_enc))  # (B, e_text) -> (B, e_comm)
        labels = torch.arange(B).to(img.device)

        img_enc_common = F.normalize(img_enc_common, dim=-1)
        text_enc_common = F.normalize(text_enc_common, dim=-1)

        logits_img2text = img_enc_common @ text_enc_common.T * torch.exp(self.t)  # (B, B)
        logits_text2img = text_enc_common @ img_enc_common.T * torch.exp(self.t)  # (B, B)

        output_dict = {
            "logits_img2text": logits_img2text,
            "logits_text2img": logits_text2img,
            "labels": labels,
            "img_enc_common": img_enc_common,
            "text_enc_common": text_enc_common,
        }
        return output_dict
