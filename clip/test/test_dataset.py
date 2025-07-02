import os
import sys

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from base_config import config
from dataset.flickr8k_dataset import Flickr8kDataset


def test_Flickr8kDataset(skip=True):
    if skip:
        assert True
    dataset = Flickr8kDataset(config)
    print(f"dataset len {len(dataset)}")
    assert len(dataset) > 0


def test_adjust_token_len():
    dataset = Flickr8kDataset(config)
    context_len = config["MODEL"]["context_len"]

    # len(tokens) > context_len
    tokens = torch.randint(low=10, high=100, size=(context_len + 10,)).tolist()
    adjusted_tokens = dataset.adjust_token_len(tokens)
    assert len(adjusted_tokens) == context_len

    # len(tokens) < context_len
    tokens = torch.randint(low=10, high=100, size=(context_len - 10,)).tolist()
    adjusted_tokens = dataset.adjust_token_len(tokens)
    assert len(adjusted_tokens) == context_len

    # len(tokens) == context_len
    tokens = torch.randint(low=10, high=100, size=(context_len,)).tolist()
    adjusted_tokens = dataset.adjust_token_len(tokens)
    assert len(adjusted_tokens) == context_len


if __name__ == "__main__":
    test_adjust_token_len()
    print("All tests passed!")
