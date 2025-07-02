import tiktoken
import torch.nn as nn


class Tokenizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = tiktoken.get_encoding("gpt2")
        self.sos_token_id = self.enc.encode("<|startoftext|>")[0]
        self.eos_token_id = self.enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

    def encode(self, x):
        return self.enc.encode(x)

    def decode(self, x):
        return self.enc.decode()

    def get_vocab_size(self):
        return self.enc.n_vocab
