from pathlib import Path

import torch
from model.tokenizer import Tokenizer
from torch.utils.data import Dataset


class DatasetBase(Dataset):
    def __init__(self, cfg, mode="train"):
        self.cfg = cfg
        self.mode = mode
        self.root_dir = Path(cfg["DATA"]["root_dir"])
        self.tokenizer = Tokenizer()
        self.context_len = cfg["MODEL"]["text_encoder"]["context_len"]

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    def process_text(self, caption):
        caption_tokenized = self.tokenize_text(caption)
        caption_tokenized = self.adjust_token_len(caption_tokenized)
        caption_tokenized = torch.tensor(caption_tokenized)
        eos_id = self.find_eos_token(caption_tokenized)
        return caption_tokenized, eos_id

    def tokenize_text(self, text):
        tokens = self.tokenizer.encode(text)
        return tokens

    def adjust_token_len(self, tokens):
        max_tokens = self.context_len - 2
        tokens = tokens[:max_tokens]

        tokens = [self.tokenizer.sos_token_id] + tokens + [self.tokenizer.eos_token_id]

        if len(tokens) < self.context_len:
            tokens += [0] * (self.context_len - len(tokens))

        return tokens

    def find_eos_token(self, tokens):
        indices = (tokens == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
        return indices[0]
