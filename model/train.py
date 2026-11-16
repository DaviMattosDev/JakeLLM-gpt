"""Train JakeLLM on a local text file or an opt-in Portuguese Wikipedia sample."""

import argparse
import math
import random
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model.gpt_model import GPT


class TextDataset(Dataset):
    """Non-overlapping windows of x/y pairs; y is shifted one token ahead."""

    def __init__(self, token_ids: list[int], block_size: int = 128):
        if block_size < 1:
            raise ValueError("block_size must be positive")
        self.token_ids = token_ids
        self.block_size = block_size
        self.starts = list(range(0, len(token_ids) - block_size, block_size))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int):
        i = self.starts[index]
        window = self.token_ids[i : i + self.block_size + 1]
        return torch.tensor(window[:-1]), torch.tensor(window[1:])


def load_texts(path: str, wikipedia: bool, max_docs: int) -> list[str]:
    if wikipedia:
        from datasets import load_dataset  # Only required for the opt-in online source.

        ds = load_dataset("wikimedia/wikipedia", "20231101.pt", split="train[:1%]")
        return [row["text"] for row in ds.select(range(min(len(ds), max_docs))) if row["text"].strip()]
    return [Path(path).read_text(encoding="utf-8")]


def evaluate(model: GPT, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss, batches = 0.0, 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
            batches += 1
    return total_loss / batches


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train JakeLLM (causal language model)")
    p.add_argument("--corpus", default="data/raw/corpus.txt")
    p.add_argument("--wikipedia", action="store_true", help="Download a Portuguese Wikipedia sample")
    p.add_argument("--max-docs", type=int, default=5000)
    p.add_argument("--tokenizer-dir", default="tokenizer")
    p.add_argument("--checkpoint", default="checkpoints/jakellm.pt")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--max-steps", type=int, default=0, help="Max batches per epoch, 0 = all")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return p.parse_args(argv)


def main(argv=None):
    from tokenizers import ByteLevelBPETokenizer

    args = parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.max_docs < 1 or args.lr <= 0:
        raise ValueError("epochs, batch-size, max-docs and lr must be positive")
    if not 0 < args.val_ratio < 0.5:
        raise ValueError("val-ratio must be between 0 and 0.5")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    vocab = Path(args.tokenizer_dir) / "vocab.json"
    merges = Path(args.tokenizer_dir) / "merges.txt"
    if not vocab.is_file() or not merges.is_file():
        raise FileNotFoundError("Tokenizer not found. Run: python -m tokenizer.train_tokenizer")
    tokenizer = ByteLevelBPETokenizer.from_file(str(vocab), str(merges))

    texts = load_texts(args.corpus, args.wikipedia, args.max_docs)
    eos = tokenizer.token_to_id("<eos>")
    ids = []
    for text in texts:
        if text.strip():
            ids.extend(tokenizer.encode(text).ids)
            if eos is not None:
                ids.append(eos)
    if len(ids) < 2 * (args.block_size + 1):
        raise ValueError("Not enough tokens for train/validation; reduce --block-size or expand the corpus")

    # Split the token stream before creating windows to prevent overlap/leakage.
    val_count = max(args.block_size + 1, round(len(ids) * args.val_ratio))
    train_ids, val_ids = ids[:-val_count], ids[-val_count:]
    train_data = TextDataset(train_ids, args.block_size)
    val_data = TextDataset(val_ids, args.block_size)
    if not train_data or not val_data:
        raise ValueError("Empty train or validation set; expand the corpus")
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    model = GPT(
        vocab_size=tokenizer.get_vocab_size(), dim=args.dim, layers=args.layers,
        heads=args.heads, block_size=args.block_size, dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, steps = 0.0, 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            steps += 1
            if args.max_steps and steps >= args.max_steps:
                break
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: train_loss={total_loss / steps:.4f} "
              f"val_loss={val_loss:.4f} val_ppl={math.exp(min(val_loss, 20)):.2f}")
        if val_loss < best_loss:
            best_loss = val_loss
            output = Path(args.checkpoint)
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "format_version": 2, "model_config": model.config,
                "model_state_dict": model.state_dict(), "epoch": epoch,
                "val_loss": val_loss,
            }, output)
            print(f"Saved best checkpoint to {output}")


if __name__ == "__main__":
    main()
