import logging
import os
from datetime import datetime
from pathlib import Path

import torch
import yaml


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
        now = ""
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
