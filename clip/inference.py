import argparse

from dataset.flickr8k_dataset import Flickr8kDataset
from model.clip_model import CLIPModel
from utils.misc import get_logger, load_config, make_artifacts_dirs
from utils.trainer import Trainer


def inference(args):
    config = load_config(args.config)
    config = make_artifacts_dirs(config, log_datetime=True)
    logger = get_logger(config["LOG_DIR"])
    trainer = Trainer(config, logger)

    train_dataset = Flickr8kDataset(cfg=config, mode="train")
    val_dataset = Flickr8kDataset(cfg=config, mode="val")

    model = CLIPModel(config)
    trainer.set_model(model)

    trainer.set_dataset(train_dataset, val_dataset, data_config=config["DATA"], val_set_batch_size=1)
    trainer.load_checkpoint(args.ckpt)
    trainer.query_data_from_prompt(args.prompt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/gpt_config.yaml", help="Config path")
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--prompt", type=str)
    args = parser.parse_args()
    inference(args)
