import torch
import torch.nn.functional as F
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
            if epoch % self.config["OPTIM"]["save_ckpt_every"] == 0:
                self.save_checkpoint()

    def train_one_epoch(self):
        self.model.train()
        self.reshuffle_dataloader()

        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for n_iter, (img, text, eos_id) in pbar:
            img = img.to(self.device)
            text = text.to(self.device)
            eos_id = eos_id.to(self.device)

            logits_img2text, logits_text2img, labels = self.model(img, text, eos_id)
            loss, loss_dict = self.loss_fn(logits_img2text, logits_text2img, labels)
            cos_sim = self.get_cos_sim(logits_img2text, logits_text2img)
            loss_dict["cos_sim"] = cos_sim.item()
            self.write_dict_to_tb(loss_dict, self.total_iters, prefix="train")

            loss.backward()
            self.gradient_clip()
            self.accumulate_gradients()

            self.total_iters += 1
            pbar.set_postfix(
                {
                    "mode": "train",
                    "epoch": f"{self.epoch}/{self.config['OPTIM']['num_epochs']}",
                    "loss": loss.item(),
                    "cos_sim": cos_sim.item(),
                }
            )
        pbar.close()

    @torch.no_grad()
    def evaluate_model(self):
        self.model.eval()
        eval_loss = []
        eval_cos_sim = []
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
            cos_sim = self.get_cos_sim(logits_img2text, logits_text2img)
            eval_cos_sim.append(cos_sim)
            pbar.set_postfix(
                {
                    "mode": "eval",
                    "epoch": f"{self.epoch}/{self.config['OPTIM']['num_epochs']}",
                    "eval_loss": loss.item(),
                    "eval_cos_sim": cos_sim.item(),
                }
            )

        pbar.close()

        eval_loss = torch.tensor(eval_loss).mean()
        eval_cos_sim = torch.tensor(eval_cos_sim).mean()
        self.logger.info(
            f"Epoch {self.epoch}/{self.num_epochs}, val_loss: {torch.tensor(eval_loss)}, val_cos_sim: {eval_cos_sim}"
        )
        self.write_float_to_tb(eval_loss, "val/loss", self.epoch)
        self.write_float_to_tb(eval_cos_sim, "val/cos_sim", self.epoch)

    def get_cos_sim(self, x1, x2):
        with torch.no_grad():
            sim = F.cosine_similarity(x1, x2, dim=-1)
            sim = sim.mean()
        return sim
