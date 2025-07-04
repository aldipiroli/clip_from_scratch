import argparse

from dataset.voc_dataset import VOCDataset
from model.clip_model import CLIPModel
from utils.misc import get_logger, load_config, make_artifacts_dirs
from utils.trainer import Trainer


def inference(args):
    config = load_config(args.config)
    config = make_artifacts_dirs(config, log_datetime=False)
    logger = get_logger(config["LOG_DIR"])
    trainer = Trainer(config, logger)

    train_dataset = VOCDataset(cfg=config, mode="train")
    val_dataset = VOCDataset(cfg=config, mode="val")

    model = CLIPModel(config)
    trainer.set_model(model)

    trainer.set_dataset(
        train_dataset, val_dataset, data_config=config["DATA"], val_set_batch_size=1, shuffle_valset=True
    )
    trainer.load_checkpoint(args.ckpt)
    trainer.zero_shot_cls()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/gpt_config.yaml", help="Config path")
    parser.add_argument("--ckpt", type=str)
    args = parser.parse_args()
    inference(args)
