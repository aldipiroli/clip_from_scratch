import os
from pathlib import Path

from dataset.dataset_base import DatasetBase
from dataset.utils import download_flickr8k_dataset
from model.tokenizer import Tokenizer
from PIL import Image
from torchvision import transforms


class Flickr8kDataset(DatasetBase):
    def __init__(self, cfg, mode="train"):
        super().__init__(cfg, mode)
        self.cfg = cfg
        self.mode = mode
        self.root_dir = Path(cfg["DATA"]["root_dir"])
        self.root_dir = download_flickr8k_dataset(self.root_dir)
        self.images_dir = os.path.join(self.root_dir, "images", "Flicker8k_Dataset")
        self.captions_file = os.path.join(self.root_dir, "captions", "Flickr8k.token.txt")
        self.transform = transforms.Compose(
            [
                transforms.Resize(232, interpolation=Image.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(  # ImageNet params
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.load_samples()
        n_train = int(len(self.image_files) * 0.9)
        if self.mode == "train":
            self.image_files = self.image_files[:n_train]
        else:
            self.image_files = self.image_files[n_train:]
        self.tokenizer = Tokenizer()
        self.context_len = cfg["MODEL"]["text_encoder"]["context_len"]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        # caption = random.choice(self.caption_dict[img_name])
        caption = self.caption_dict[img_name][0]
        caption_tokenized, eos_id = self.process_text(caption)
        return image, img_name, caption_tokenized, eos_id

    def load_samples(self):
        self.caption_dict = {}
        with open(self.captions_file, "r") as f:
            for line in f:
                img_tag, caption = line.strip().split("\t")
                img_name = img_tag.split("#")[0]
                img_path = os.path.join(self.images_dir, img_name)
                if os.path.isfile(img_path):
                    self.caption_dict.setdefault(img_name, []).append(caption)
                else:
                    print(f"Skipping {img_path}")
        self.image_files = list(self.caption_dict.keys())
