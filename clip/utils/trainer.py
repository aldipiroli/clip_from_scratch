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

    def train_one_epoch(self):
        self.model.train()
        self.reshuffle_dataloader()

        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for n_iter, (img, text, eos_id) in pbar:
            # if n_iter > 50:
            #     break
            self.optimizer.zero_grad()
            img = img.to(self.device)
            text = text.to(self.device)
            eos_id = eos_id.to(self.device)

            logits_img2text, logits_text2img, labels = self.model(img, text, eos_id)
            loss, loss_dict = self.loss_fn(logits_img2text, logits_text2img, labels)
            self.write_dict_to_tb(loss_dict, self.total_iters, prefix="train")

            loss.backward()
            self.gradient_clip()
            self.optimizer.step()
            self.total_iters += 1
            pbar.set_postfix({"epoch": f"{self.epoch}/{self.config['OPTIM']['num_epochs']}", "loss": loss.item()})
        pbar.close()

    @torch.no_grad()
    def evaluate_model(self):
        self.model.eval()
        eval_loss = []
        pbar = tqdm(enumerate(self.val_loader), total=len(self.val_loader))
        for n_iter, (img, text, eos_id) in pbar:
            if n_iter > self.config["OPTIM"]["num_iterations_val"]:
                break
            img = img.to(self.device)
            text = text.to(self.device)
            eos_id = eos_id.to(self.device)

            logits_img2text, logits_text2img, labels = self.model(img, text, eos_id)
            loss, loss_dict = self.loss_fn(logits_img2text, logits_text2img, labels)
            eval_loss.append(loss)
            pbar.set_postfix({"epoch": f"{self.epoch}/{self.config['OPTIM']['num_epochs']}", "eval_loss": loss.item()})

        pbar.close()

        eval_loss = torch.tensor(eval_loss).mean()
        self.logger.info(f"Epoch {self.epoch}/{self.num_epochs}: val loss {torch.tensor(eval_loss)}")
        self.write_float_to_tb(eval_loss, "val/loss", self.epoch)
