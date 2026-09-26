import torch
from torch import nn


class QwenGraph(nn.Module):
    def __init__(self, backbone, length):
        super().__init__()
        self.backbone = backbone
        cfg = backbone.config
        self.length = length
        self.heads = cfg.num_attention_heads
        self.kv_heads = cfg.num_key_value_heads
        self.dim = cfg.head_dim
        cos, sin = backbone.rotary_emb(
            torch.zeros(1, length, cfg.hidden_size), torch.arange(length)[None]
        )
        self.register_buffer("cos", cos[:, None])
        self.register_buffer("sin", sin[:, None])
        self.register_buffer(
            "mask", torch.triu(torch.full((length, length), -10000.0), 1)[None, None]
        )

    def rotary(self, x):
        a, b = x.chunk(2, -1)
        return x * self.cos + torch.cat((-b, a), -1) * self.sin

    def hidden(self, input_ids):
        h = self.backbone.embed_tokens(input_ids)
        for layer in self.backbone.layers:
            n = layer.input_layernorm(h)
            attn = layer.self_attn
            q = (
                attn.q_proj(n)
                .reshape(1, self.length, self.heads, self.dim)
                .transpose(1, 2)
            )
            k = (
                attn.k_proj(n)
                .reshape(1, self.length, self.kv_heads, self.dim)
                .transpose(1, 2)
            )
            v = (
                attn.v_proj(n)
                .reshape(1, self.length, self.kv_heads, self.dim)
                .transpose(1, 2)
            )
            q, k = (self.rotary(attn.q_norm(q)), self.rotary(attn.k_norm(k)))
            k = k.repeat_interleave(self.heads // self.kv_heads, dim=1)
            v = v.repeat_interleave(self.heads // self.kv_heads, dim=1)
            weights = torch.softmax(
                q @ k.transpose(-2, -1) * self.dim ** (-0.5) + self.mask, dim=-1
            )
            a = (weights @ v).transpose(1, 2).reshape(1, self.length, -1)
            h = h + attn.o_proj(a)
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        h = self.backbone.norm(h)
        return h

    def forward(self, input_ids, leaf_map):
        return leaf_map @ self.hidden(input_ids)
