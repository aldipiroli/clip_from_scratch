import os
import sys

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model.clip_model import AttentionLayer, ImgEncoder, TextEncoder, Tokenizer, TransformerLayer


def test_img_encoder():
    encoder = ImgEncoder()
    B, C, H, W = 2, 3, 224, 224
    x = torch.randn(B, C, H, W)
    patch_size = 16
    out = encoder(x)
    assert out.shape == (B, H // patch_size * W // patch_size, patch_size**2 * 3)


def test_attention_layer():
    embed_size = 64
    n_heads = 4
    layer = AttentionLayer(embed_size=64, embed_size_h=embed_size // n_heads, dropout=0.1)
    B, T, C = 2, 32, embed_size

    x = torch.randn(B, T, C)
    out = layer(x)
    assert out.shape == (B, T, embed_size // n_heads)


def test_transformer_layer():
    embed_size = 64
    n_heads = 4
    layer = TransformerLayer(embed_size=embed_size, n_heads=n_heads, dropout=0.1)
    B, T, C = 2, 32, embed_size

    x = torch.randn(B, T, C)
    out = layer(x)
    assert out.shape == (B, T, embed_size)


def test_text_encoder():
    tok = Tokenizer()
    B = 2
    embed_size = 64
    n_heads = 4
    context_len = 32
    n_layers = 8
    dropout = 0.1
    vocab_size = tok.get_vocab_size()

    text = "".join(["hello " * 100])
    tokens = tok.encode(text)
    tokens = torch.tensor(tokens).unsqueeze(0).expand(B, -1)
    tokens = tokens[:, :context_len]

    text_encoder = TextEncoder(context_len, embed_size, vocab_size, n_heads, n_layers, dropout)
    out = text_encoder(tokens)
    assert out.shape == (B, context_len, vocab_size)


if __name__ == "__main__":
    print("All tests passed!")
