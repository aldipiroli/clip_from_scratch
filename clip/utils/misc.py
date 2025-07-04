import logging
import os
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml
from PIL import Image


def get_device():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    return device


def to_device(x):
    x = x.to(get_device())
    return x


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_artifacts_dirs(cfg, log_datetime=False):
    if log_datetime:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        now = "logs/"
    cfg_artifacts = Path(cfg["ARTIFACTS_DIR"])
    cfg_artifacts = cfg_artifacts / now
    cfg["IMG_OUT_DIR"] = cfg_artifacts / "imgs"
    os.makedirs(cfg["IMG_OUT_DIR"], exist_ok=True)

    cfg["LOG_DIR"] = cfg_artifacts / "logs"
    os.makedirs(cfg["LOG_DIR"], exist_ok=True)

    cfg["TB_LOG_DIR"] = cfg_artifacts / "tb_logs"
    os.makedirs(cfg["TB_LOG_DIR"], exist_ok=True)

    cfg["CKPT_DIR"] = cfg_artifacts / "ckpts"
    os.makedirs(cfg["CKPT_DIR"], exist_ok=True)

    cfg["EMBED_DIR"] = cfg_artifacts / "embeddings"
    os.makedirs(cfg["EMBED_DIR"], exist_ok=True)
    return cfg


def get_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log.log")
    logger = logging.getLogger(f"logger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_file, mode="a")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def save_pickle(data, path):
    with open(path, "wb") as file:
        pickle.dump(data, file)


def load_pickle(path):
    with open(path, "rb") as file:
        loaded_data = pickle.load(file)
    return loaded_data


def plot_images_with_values(image_paths, values, prompt, save_dir):
    num_images = len(image_paths)
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))

    for ax, img_path, value in zip(axes, image_paths, values):
        img = Image.open(img_path)
        img = img.resize((224, 224))
        ax.imshow(img)
        ax.set_title(f"Score: {value:.2f}", fontsize=18)
        ax.axis("off")
    plt.suptitle(f"Prompt: '{prompt}'", fontsize=24)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(os.path.join(save_dir, "matches.png"))
