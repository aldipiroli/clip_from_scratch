import os
import sys

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_config import config
from dataset.flickr8k_dataset import Flickr8kDataset


def test_Flickr8kDataset(skip=True):
    if skip:
        assert True
    dataset = Flickr8kDataset(config)
    print(f"dataset len {len(dataset)}")
    assert len(dataset) > 0


if __name__ == "__main__":
    test_Flickr8kDataset(skip=False)
    print("All tests passed!")
