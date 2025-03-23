from config import config
import torch
from collections import Counter
import jieba


class DataProcessor:
    def __init__(self, file_name):
        with open(file_name, 'r', encoding='GBK') as f:
            text = f.read()

        words = list(jieba.cut(text))
        word_counts = Counter(words)
        vocab = [word for word, _ in word_counts.most_common(config.max_vocab - 1)]
        vocab.append('<UNK>')

        self.vocab_size = len(vocab)
        self.stoi = {word: i for i, word in enumerate(vocab)}
        self.itos = {i: word for i, word in enumerate(vocab)}

        data = [self.stoi.get(w, self.stoi['<UNK>']) for w in words]
        data = torch.tensor(data, dtype=torch.long, device=config.device)

        n = int(0.9 * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def encode(self, s):
        words = jieba.cut(s)
        return [self.stoi.get(w, self.stoi['<UNK>']) for w in words]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])

    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
        x = torch.stack([data[i:i + config.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + config.block_size + 1] for i in ix])
        return x.to(config.device), y.to(config.device)
