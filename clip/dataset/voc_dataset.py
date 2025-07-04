import os
from pathlib import Path

import torch
from dataset.dataset_base import DatasetBase
from PIL import Image
from torchvision import transforms
from torchvision.datasets import VOCDetection


class VOCDataset(DatasetBase):
    def __init__(self, cfg, mode):
        super().__init__(cfg, mode)
        self.cfg = cfg
        self.mode = mode
        self.root_dir = Path(cfg["DATA"]["root_dir"])
        dest_path = os.path.join(self.root_dir, "VOCdevkit/VOC2012")
        self.dataset = VOCDetection(
            root=self.root_dir, year="2012", image_set=mode, download=True if not os.path.exists(dest_path) else False
        )
        self.transforms = transforms.Compose(
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
        self.simple_transform = transforms.Compose(
            [transforms.Resize(232, interpolation=Image.BICUBIC), transforms.CenterCrop(224), transforms.ToTensor()]
        )
        self.class_to_idx = self._build_class_map()

    def _build_class_map(self):
        self.classes = [
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        ]
        return {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        objs = target["annotation"]["object"]
        if not isinstance(objs, list):
            objs = [objs]

        labels = []
        for obj in objs:
            labels.append(self.class_to_idx[obj["name"]])

        labels = torch.tensor(labels, dtype=torch.long)
        target = {"labels": labels, "image_id": torch.tensor([idx])}
        img_tr = self.transforms(img)
        return self.simple_transform(img), img_tr, target
