import os
import random
from pathlib import Path

from dataset.utils import download_flickr8k_dataset
from model.tokenizer import Tokenizer
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class Flickr8kDataset(Dataset):
    def __init__(self, cfg, mode="train"):
        self.cfg = cfg
        self.mode = mode
        self.root_dir = Path(cfg["DATA"]["root_dir"])
        self.root_dir = download_flickr8k_dataset(self.root_dir)
        self.images_dir = os.path.join(self.root_dir, "images", "Flicker8k_Dataset")
        self.captions_file = os.path.join(self.root_dir, "captions", "Flickr8k.token.txt")
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
        self.load_samples()
        n_train = int(len(self.image_files) * 0.9)
        if self.mode == "train":
            self.image_files = self.image_files[:n_train]
        else:
            self.image_files = self.image_files[n_train:]
        self.tokenizer = Tokenizer()
        self.context_len = cfg["MODEL"]["context_len"]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        caption = random.choice(self.caption_dict[img_name])
        return image, caption

    def load_samples(self):
        self.caption_dict = {}
        with open(self.captions_file, "r") as f:
            for line in f:
                img_tag, caption = line.strip().split("\t")
                img_name = img_tag.split("#")[0]
                self.caption_dict.setdefault(img_name, []).append(caption)
        self.image_files = list(self.caption_dict.keys())

    def tokenize_text(self, text):
        tokens = self.tokenizer.encode(text)
        return tokens

    def adjust_token_len(self, tokens):
        if len(tokens) + 2 >= self.context_len:
            tokens = tokens[: self.context_len - 2]
            tokens.insert(0, self.tokenizer.sos_token_id)
            tokens.insert(-1, self.tokenizer.eos_token_id)
        elif len(tokens) + 2 < self.context_len:
            tokens.insert(0, self.tokenizer.sos_token_id)
            tokens.insert(-1, self.tokenizer.eos_token_id)
            tokens += [0] * (self.context_len - len(tokens))
        else:
            raise ValueError
        return tokens
