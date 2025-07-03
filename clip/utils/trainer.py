import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.misc import plot_images_with_values
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

            output_dict = self.model(img, text, eos_id)
            loss, loss_dict = self.loss_fn(
                output_dict["logits_img2text"], output_dict["logits_text2img"], output_dict["labels"]
            )
            cos_sim = self.get_cos_sim(output_dict["logits_img2text"], output_dict["logits_text2img"])
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
        for n_iter, (img, _, text, eos_id) in pbar:
            if n_iter > self.config["OPTIM"]["num_iterations_val"]:
                break
            img = img.to(self.device)
            text = text.to(self.device)
            eos_id = eos_id.to(self.device)

            output_dict = self.model(img, text, eos_id)
            loss, loss_dict = self.loss_fn(
                output_dict["logits_img2text"], output_dict["logits_text2img"], output_dict["labels"]
            )
            eval_loss.append(loss)
            cos_sim = self.get_cos_sim(output_dict["logits_img2text"], output_dict["logits_text2img"])
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

    def query_data_from_prompt(self, prompt):
        self.model.eval()
        prompt_tokenized, eos_id = self.val_dataset.process_text(prompt)
        prompt_tokenized = prompt_tokenized.unsqueeze(0)
        all_cos_sim = []
        all_img_names = []

        pbar = tqdm(enumerate(self.val_loader), total=len(self.val_loader))
        for n_iter, (img, img_name, caption, _) in pbar:
            if n_iter > 100:
                break
            img = img.to(self.device)
            prompt_tokenized = prompt_tokenized.to(self.device)
            eos_id = eos_id.to(self.device)
            output_dict = self.model(img, prompt_tokenized, eos_id)
            cos_sim = self.get_cos_sim(output_dict["img_enc_common"], output_dict["text_enc_common"])
            self.logger.info(f"cos_sim {cos_sim}")
            all_cos_sim.append(cos_sim)
            img_path = os.path.join(self.val_dataset.images_dir, img_name[0])
            assert os.path.isfile(img_path)
            all_img_names.append(img_path)

        all_cos_sim = torch.tensor(all_cos_sim)
        val, ind = torch.topk(all_cos_sim, 5)
        selected_imgs = [all_img_names[i] for i in ind.tolist()]
        selected_cos_sim = [all_cos_sim[i] for i in ind.tolist()]
        plot_images_with_values(selected_imgs, selected_cos_sim, prompt=prompt, save_dir=self.config["IMG_OUT_DIR"])
        # plot_images_with_values(all_img_names, all_cos_sim, prompt=prompt, save_dir=self.config["IMG_OUT_DIR"])
