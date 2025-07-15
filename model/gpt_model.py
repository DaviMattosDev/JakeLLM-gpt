import torch
import torch.nn as nn

class GPTBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.mlp(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, dim=256, layers=4, heads=4):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(512, dim)
        self.blocks = nn.Sequential(*[GPTBlock(dim, heads) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        B, T = x.size()
        pos = torch.arange(0, T).unsqueeze(0).to(x.device)
        x = self.token_emb(x) + self.pos_emb(pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)