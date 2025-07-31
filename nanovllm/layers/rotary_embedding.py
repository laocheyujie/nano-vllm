from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    # 将 cos 和 sin 扩展到与 x 相同的维度 [max_position_embeddings, 1, dim/2]
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    # 将 - 和 + 拆卡再合并，详见 RoPE 笔记
    x1, x2 = torch.chunk(x.to(torch.float32), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        # inv_freq 是频率的倒数 [dim/2]
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        # t [max_position_embeddings]
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        # 用 einsum（爱因斯坦求和约定）来高效构造一个二维矩阵，它等价于矩阵的外积
        # 即 freqs[i][j] = t[i] * inv_freq[j]
        # [max_position_embeddings, dim/2]
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        # cos, sin [max_position_embeddings, dim/2]
        cos = freqs.cos()
        sin = freqs.sin()
        # 将 cos 和 sin 拼接起来，[max_position_embeddings, dim]
        cache = torch.cat((cos, sin), dim=-1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = positions.size(0)
        cos_sin = self.cos_sin_cache[positions]
        # 将 cos 和 sin 拆分出来
        cos, sin = cos_sin.chunk(2, dim=-1)
        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query = apply_rotary_emb(query, cos, sin).view(query_shape)
        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key = apply_rotary_emb(key, cos, sin).view(key_shape)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    assert rope_scaling is None
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
