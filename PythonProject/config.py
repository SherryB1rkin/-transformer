class Config:
    device = "cuda"

    # 模型参数
    vocab_size = None
    block_size = 256
    n_embd = 512
    n_head = 8
    head_size = int(n_embd / n_head)
    n_layer = 6
    dropout = 0.2
    bias = False

    # 训练参数
    batch_size = 64
    learning_rate = 3e-4
    max_iters = 5000
    warmup_iters = 50
    lr_decay_iters = 500
    min_lr = 1e-5

    # 生成参数
    temperature = 0.8
    top_k = 200
    #eval_interval = int(max_iters / 10)
    max_gen_token = 200

config = Config()
