import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.misc import load_pickle, plot_image_and_text, plot_images_with_values, save_pickle
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
        for n_iter, (img, _, text, eos_id) in pbar:
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

    def compute_dataset_embeddings(self):
        self.model.eval()
        all_infos = []

        pbar = tqdm(enumerate(self.val_loader), total=len(self.val_loader))
        for n_iter, (img, img_name, caption, eos_id) in pbar:
            img = img.to(self.device)
            caption = caption.to(self.device)
            eos_id = eos_id.to(self.device)
            output_dict = self.model(img, caption, eos_id)
            img_path = Path(self.val_dataset.images_dir) / img_name[0]
            assert os.path.isfile(img_path)
            output_dict["img_name"] = img_path
            all_infos.append(output_dict)

        save_path = Path(self.config["EMBED_DIR"]) / "embeddings.pkl"
        save_pickle(all_infos, save_path)
        self.logger.info(f"Saved embeddings in {save_path}")

    def prepare_embeddings(self, all_infos):
        embeds = []
        for info in all_infos:
            embeds.append(info["img_enc_common"])
        embeds = torch.cat(embeds, 0)
        return embeds

    def get_prompt_embed(self, prompt):
        self.model.eval()
        prompt_tokenized, eos_id = self.val_dataset.process_text(prompt)
        prompt_tokenized = prompt_tokenized.unsqueeze(0)
        prompt_tokenized = prompt_tokenized.to(self.device)
        eos_id = eos_id.to(self.device)
        img_size = self.config["MODEL"]["img_size"]
        img = torch.randn(1, img_size[2], img_size[1], img_size[0]).to(self.device)

        output_dict = self.model(img, prompt_tokenized, eos_id)
        text_enc_common = output_dict["text_enc_common"]
        return text_enc_common

    def query_data_from_prompt(self, prompt, embed_path=None, query_numbers=5):
        if embed_path is None:
            self.compute_dataset_embeddings()
            embed_path = Path(self.config["EMBED_DIR"]) / "embeddings.pkl"

        all_infos = load_pickle(Path(embed_path))
        embeds = self.prepare_embeddings(all_infos)
        self.logger.info(f"Loaded embeddings: {embed_path} with size: {embeds.shape}")

        text_enc_common = self.get_prompt_embed(prompt)
        cos_sim = torch.mm(embeds, text_enc_common.permute(1, 0)).squeeze(-1)
        val, ind = torch.topk(cos_sim, query_numbers)

        selected_imgs = [all_infos[i]["img_name"] for i in ind.tolist()]
        selected_cos_sim = [cos_sim[i] for i in ind.tolist()]
        plot_images_with_values(selected_imgs, selected_cos_sim, prompt=prompt, save_dir=self.config["IMG_OUT_DIR"])

    def generate_zero_shot_text_queries(self):
        possible_prompts = []
        for curr_class in self.val_dataset.classes:
            prompt = f"A photo of a {curr_class}"
            possible_prompts.append(prompt)
        return possible_prompts

    def zero_shot_cls(self):
        self.model.eval()
        possible_prompts = self.generate_zero_shot_text_queries()
        for img_id, (img_no_tr, img, _) in enumerate(self.val_loader):
            if img_id > 100:
                break
            all_cos_sim = []
            for n_iter, (prompt) in enumerate(possible_prompts):
                img = img.to(self.device)
                prompt_tok, eos_id = self.val_dataset.process_text(prompt)
                prompt_tok = prompt_tok.unsqueeze(0).to(self.device)
                eos_id = eos_id.unsqueeze(0).to(self.device)
                output_dict = self.model(img, prompt_tok, eos_id)
                cos_sim = self.get_cos_sim(output_dict["img_enc_common"], output_dict["text_enc_common"])
                all_cos_sim.append(cos_sim.detach().cpu().numpy())

            sorted_indices = sorted(range(len(all_cos_sim)), key=lambda i: all_cos_sim[i], reverse=True)
            all_cos_sim = [all_cos_sim[i] for i in sorted_indices]
            possible_prompts = [possible_prompts[i] for i in sorted_indices]
            possible_classes = [self.val_dataset.classes[i] for i in sorted_indices]
            plot_image_and_text(
                img_no_tr[0], possible_classes, all_cos_sim, save_dir=self.config["IMG_OUT_DIR"], img_id=img_id
            )
            self.logger.info(f"Processed {img_id}/{len(self.val_loader)}")
