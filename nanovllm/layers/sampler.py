import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.to(torch.float)
        greedy_tokens = logits.argmax(dim=-1)
        logits.div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)
        epsilon = 1e-10
        # 用 probs 概率除以一个随机扰动项（指数分布 + epsilon），会打乱原始的概率排序
        # 在扰动之后的值上取最大值，相当于对打乱排序后的概率进行 argmax，就形成了采样
        # 背后的原理：Gumbel-Max Trick
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1) + epsilon).argmax(dim=-1)  
        return torch.where(temperatures == 0, greedy_tokens, sample_tokens)
