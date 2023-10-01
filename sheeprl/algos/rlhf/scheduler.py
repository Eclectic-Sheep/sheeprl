import math


class CosineSchedulerWithWarmup:
    def __init__(self, lr: float, warmup_steps: int, lr_decay_steps: int, min_lr: float = 1e-8):
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.lr_decay_steps = lr_decay_steps
        self.min_lr = min_lr

    def get_lr(self, it: int):
        # 1) linear warmup for warmup_iters steps
        if it < self.warmup_steps:
            return self.lr * it / self.warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.lr_decay_steps:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.warmup_steps) / (self.lr_decay_steps - self.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.min_lr + coeff * (self.lr - self.min_lr)
