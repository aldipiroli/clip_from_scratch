import os
import sys

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model.clip_model import ImgEncoder


def test_img_encoder():
    encoder = ImgEncoder()
    B, C, H, W = 2, 3, 224, 224
    x = torch.randn(B, C, H, W)
    patch_size = 16
    out = encoder(x)
    assert out.shape == (B, H // patch_size * W // patch_size, patch_size**2 * 3)


if __name__ == "__main__":
    print("All tests passed!")
