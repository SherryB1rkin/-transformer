import random
import textwrap
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import optim
from model import Transformer
from process import DataProcessor
from config import config


def main():
    file_name="射雕三部曲.txt"
    processor = DataProcessor(file_name, block_size=config.block_size)
    config.vocab_size = processor.vocab_size

    model = Transformer(config).to(config.device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)

    train_losses = []
    val_losses = []
    iterations = []

    def get_lr(it):
        if it < config.warmup_iters:
            return config.learning_rate * it / config.warmup_iters
        if it > config.lr_decay_iters:
            return config.min_lr
        decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
        return config.min_lr + 0.5 * (config.learning_rate - config.min_lr) * (1 + np.cos(np.pi * decay_ratio))

    for iter in range(config.max_iters):
        lr = get_lr(iter)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        x, y = processor.get_batch('train')
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter % 100 == 0 or iter == config.max_iters - 1:
            val_x, val_y = processor.get_batch('val')
            with torch.no_grad():
                _, val_loss = model(val_x, val_y)

            train_losses.append(loss.item())
            val_losses.append(val_loss.item())
            iterations.append(iter)

            print(f"Iter {iter:04d} | Train Loss {loss.item():.4f} | Val Loss {val_loss.item():.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, train_losses, label='Training Loss', marker='o')
    plt.plot(iterations, val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Training Iterations')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)

    plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
    plt.show()

    max_start = len(processor.val_data) - config.block_size - config.max_new_tokens
    start_idx = random.randint(0, max_start)

    context = processor.val_data[start_idx: start_idx + config.block_size]
    context_tensor = torch.tensor(context, dtype=torch.long, device=config.device)

    real_next = processor.val_data[start_idx + config.block_size: start_idx + config.block_size + config.max_new_tokens]

    with torch.no_grad():
        generated = model.generate(
            context_tensor,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_k=config.top_k
        )

    def format_text(tokens):
        text = processor.decode(tokens)
        return textwrap.fill(text, width=config.wrap_width)

    print("=" * 50 + "\n上文内容：")
    print(format_text(context))
    print("\n" + "=" * 50 + "\n生成内容：")
    print(format_text(generated[0].tolist()))
    print("\n" + "=" * 50 + "\n原文后续：")
    print(format_text(real_next))


if __name__ == "__main__":
    main()


