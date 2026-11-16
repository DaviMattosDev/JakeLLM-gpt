"""A small causal decoder-only Transformer for next-token prediction."""

import torch
from torch import nn


class GPTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # True means "do not attend". Future tokens must never be visible.
        t = x.size(1)
        causal_mask = torch.ones(t, t, device=x.device, dtype=torch.bool).triu(1)
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = x + attn_out
        return x + self.mlp(self.norm2(x))


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int = 256,
        layers: int = 4,
        heads: int = 4,
        block_size: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        if min(vocab_size, dim, layers, heads, block_size) <= 0:
            raise ValueError("Model dimensions must be positive")
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        self.config = dict(
            vocab_size=vocab_size, dim=dim, layers=layers, heads=heads,
            block_size=block_size, dropout=dropout,
        )
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(block_size, dim)
        self.emb_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([GPTBlock(dim, heads, dropout) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("Expected token IDs of shape [batch, time]")
        _, t = token_ids.shape
        if not 1 <= t <= self.block_size:
            raise ValueError(f"Sequence length must be between 1 and {self.block_size}; got {t}")
        positions = torch.arange(t, device=token_ids.device)
        x = self.emb_dropout(self.token_emb(token_ids) + self.pos_emb(positions))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 20,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Generate from a single prompt. Temperature 0 enables greedy decoding."""
        if input_ids.ndim != 2 or input_ids.size(0) != 1 or input_ids.size(1) == 0:
            raise ValueError("generate expects one nonempty prompt: shape [1, time]")
        if max_new_tokens < 0 or temperature < 0 or top_k < 0:
            raise ValueError("max_new_tokens, temperature and top_k must be nonnegative")
        if max_new_tokens == 0:
            return input_ids.clone()

        was_training = self.training
        self.eval()
        try:
            ids = input_ids
            for _ in range(max_new_tokens):
                logits = self(ids[:, -self.block_size:])[:, -1, :]
                if temperature == 0:
                    next_id = logits.argmax(dim=-1, keepdim=True)
                else:
                    logits = logits / temperature
                    if top_k > 0:
                        values, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
                        logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
                    probs = torch.softmax(logits, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1)
                ids = torch.cat((ids, next_id), dim=1)
                if eos_token_id is not None and next_id.item() == eos_token_id:
                    break
            return ids
        finally:
            self.train(was_training)
