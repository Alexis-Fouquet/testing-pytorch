import torch


def seed(seed: int = 42):
    # Should always be the same
    torch.manual_seed(seed)
