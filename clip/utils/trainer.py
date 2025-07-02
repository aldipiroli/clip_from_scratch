import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.trainer_base import TrainerBase


class Trainer(TrainerBase):
    def __init__(self, config, logger):
        super().__init__(config, logger)

    def reshuffle_dataloader(self):
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config["DATA"]["batch_size"],
            shuffle=True,
        )

    def train(self):
        self.logger.info("Started training..")
        start_epoch = self.epoch
        for epoch in range(start_epoch, self.config["OPTIM"]["num_epochs"]):
            self.epoch = epoch
            self.train_one_epoch()
            self.evaluate_model()
            self.save_checkpoint()
            if epoch % self.eval_every == 0:
                self.generate_output()

    def train_one_epoch(self):
        self.model.train()
        self.reshuffle_dataloader()
        train_loss = []
        pbar = tqdm(enumerate(self.train_loader), total=self.config["OPTIM"]["num_epochs"])
        for n_iter, (img, text) in pbar:
            self.optimizer.zero_grad()
            img = img.to(self.device)
            text = text.to(self.device)

            preds = self.model(img, text)
            loss = self.loss_fn()
            train_loss.append(loss)
            self.write_float_to_tb(loss, "train/loss", self.total_iters)

            loss.backward()
            self.gradient_clip()
            self.optimizer.step()
            self.total_iters += 1
            pbar.set_postfix({"epoch": f"{self.epoch}/{self.config['OPTIM']['num_epochs']}", "loss": loss.item()})
        self.logger.info(f"Epoch {self.epoch}/{self.num_epochs}: train loss {torch.mean(torch.tensor(train_loss))}")

    @torch.no_grad()
    def evaluate_model(self):
        self.model.eval()
        eval_loss = []
        pbar = tqdm(enumerate(self.val_loader), total=self.config["OPTIM"]["num_iterations_val"])
        for n_iter, (img, text) in pbar:
            img = img.to(self.device)
            text = text.to(self.device)
            preds = self.model(img, text)
            loss = self.loss_fn()
            eval_loss.append(loss)
        eval_loss = torch.tensor(eval_loss).mean()
        self.logger.info(f"Epoch {self.epoch}/{self.num_epochs}: val loss {torch.tensor(eval_loss)}")
        self.write_float_to_tb(eval_loss, "val/loss", self.epoch)
