import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config


class Head(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.query = nn.Linear(config.n_embd, config.head_size)
        self.key = nn.Linear(config.n_embd, config.head_size)
        self.value = nn.Linear(config.n_embd, config.head_size)

        self.register_buffer("tril", torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attention_matrix = q @ k.transpose(-2, -1) / config.head_size ** -0.5
        attention_matrix = attention_matrix.masked_fill(self.tril == 0, float("-inf"))
        attention_matrix = F.softmax(attention_matrix, dim=-1)
        attention_matrix = self.dropout(attention_matrix)
        out = attention_matrix @ v
        return out


class MultiHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([Head(config) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.head_size * config.n_head, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sa = MultiHead(config)
        self.feedforward = FeedForward(config)
        self.norm1 = nn.LayerNorm(config.n_embd)
        self.norm2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.sa(self.norm1(x))
        x = x + self.feedforward(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.line = nn.Linear(config.n_embd, config.vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        batch_sz, block_sz = idx.size()

        pos = torch.arange(0, block_sz, dtype=torch.long, device=idx.device)
        token_emb = self.tok_emb(idx)  # (B,T,C)
        pos_emb = self.pos_emb(pos)  # (T,C)
        x = token_emb + pos_emb  # (B,T,C)

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.line(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, seq_tokens, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            tokens_input = seq_tokens[:, -self.config.block_size:]
            logits, _ = self.forward(tokens_input)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('inf')

            probs = F.softmax(logits, dim=-1)
            tokens_next = torch.multinomial(probs, num_samples=1).to(config.device)
            seq_tokens = torch.cat((seq_tokens, tokens_next), dim=1)

        new_tokens = seq_tokens[:, -config.max_gen_token:]
        return new_tokens
