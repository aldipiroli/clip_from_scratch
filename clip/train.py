import argparse

from dataset.flickr8k_dataset import Flickr8kDataset
from model.clip_loss import CLIPLoss
from model.clip_model import CLIPModel
from utils.misc import get_logger, load_config, make_artifacts_dirs
from utils.trainer import Trainer


def train(args):
    config = load_config(args.config)
    config = make_artifacts_dirs(config, log_datetime=True)
    logger = get_logger(config["LOG_DIR"])
    trainer = Trainer(config, logger)

    train_dataset = Flickr8kDataset(cfg=config, mode="train")
    val_dataset = Flickr8kDataset(cfg=config, mode="val")

    model = CLIPModel(config)
    trainer.set_model(model)

    trainer.set_dataset(train_dataset, val_dataset, data_config=config["DATA"])
    trainer.set_optimizer(optim_config=config["OPTIM"])
    trainer.set_loss_function(CLIPLoss())

    trainer.load_latest_checkpoint()
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, default="config/gpt_config.yaml", help="Config path")
    args = parser.parse_args()
    train(args)
