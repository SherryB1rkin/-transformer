import torch
from config import config


class DataProcessor:
    def __init__(self, file_name, block_size=256):
        with open(file_name, 'r', encoding='GBK') as f:
            text = f.read()

        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

        data = torch.tensor(self.encode(text), dtype=torch.long, device=config.device)
        n = int(0.9 * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]
        self.block_size = block_size

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])

    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        start = torch.randint(len(data) - self.block_size, (config.batch_size,))
        x = torch.stack([data[i:i + self.block_size].long() for i in start])
        y = torch.stack([data[i + 1:i + self.block_size + 1].long() for i in start])
        return x.to(config.device), y.to(config.device)
