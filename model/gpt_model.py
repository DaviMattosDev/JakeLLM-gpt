import torch
import torch.nn as nn


class GPTBlock(nn.Module):
    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # True blocks attention to future tokens, including the training target.
        mask = torch.ones(x.size(1), x.size(1), device=x.device, dtype=torch.bool).triu(1)
        attn_out, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        x = self.norm1(x + attn_out)
        return self.norm2(x + self.mlp(x))


class GPT(nn.Module):
    def __init__(self, vocab_size, dim=256, layers=4, heads=4,
                 block_size=512, dropout=0.1):
        super().__init__()
        if min(vocab_size, dim, layers, heads, block_size) < 1 or dim % heads:
            raise ValueError('Model sizes must be positive and dim divisible by heads')
        self.config = dict(vocab_size=vocab_size, dim=dim, layers=layers,
                           heads=heads, block_size=block_size, dropout=dropout)
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(block_size, dim)
        self.blocks = nn.Sequential(*[GPTBlock(dim, heads, dropout) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        if x.ndim != 2 or not 0 < x.size(1) <= self.block_size:
            raise ValueError(f'Expected [batch, tokens] with 1..{self.block_size} tokens')
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = self.token_emb(x) + self.pos_emb(pos)
        return self.head(self.ln_f(self.blocks(x)))
