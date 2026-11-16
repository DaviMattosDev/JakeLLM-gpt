"""Run from the repository root: python -m model.train --help."""
import argparse
import hashlib
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm

from model.gpt_model import GPT

ROOT = Path(__file__).resolve().parents[1]


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size=128):
        if block_size < 1:
            raise ValueError('block_size must be positive')
        self.examples = []
        for text in texts:
            tokens = tokenizer.encode(text).ids
            # Each example needs one extra token for the final prediction target.
            for start in range(0, len(tokens) - block_size, block_size):
                self.examples.append(torch.tensor(tokens[start:start + block_size + 1], dtype=torch.long))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch[:, :-1])
        targets = batch[:, 1:]
        total += F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction='sum').item()
        count += targets.numel()
    return total / count


def main():
    parser = argparse.ArgumentParser(description='Train a causal Portuguese mini-GPT')
    parser.add_argument('--train-file', type=Path, help='Local UTF-8 corpus, one document per nonempty line')
    parser.add_argument('--validation-file', type=Path, help='Separate local validation corpus')
    parser.add_argument('--tokenizer-dir', type=Path, default=ROOT / 'tokenizer')
    parser.add_argument('--output', type=Path, default=ROOT / 'model_weights.pth')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--block-size', type=int, default=128)
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-documents', type=int, default=5000)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.block_size, args.max_documents) < 1 or not args.lr > 0:
        parser.error('epochs, batch-size, block-size, max-documents and lr must be positive')
    if args.validation_file and not args.train_file:
        parser.error('--validation-file requires --train-file')
    torch.manual_seed(args.seed)
    tokenizer = ByteLevelBPETokenizer(str(args.tokenizer_dir / 'vocab.json'), str(args.tokenizer_dir / 'merges.txt'))
    if args.train_file:
        texts = [line.strip() for line in args.train_file.read_text(encoding='utf-8').splitlines() if line.strip()]
        texts = texts[:args.max_documents]
    else:
        from datasets import load_dataset
        dataset = load_dataset('wikimedia/wikipedia', '20231101.pt', split='train[:1%]')
        texts = dataset['text'][:args.max_documents]
    # Split documents before chunking to avoid putting adjacent chunks in validation.
    if args.validation_file:
        validation = [line.strip() for line in args.validation_file.read_text(encoding='utf-8').splitlines() if line.strip()]
    else:
        if len(texts) < 2:
            parser.error('Provide at least two documents or a separate validation file')
        order = torch.randperm(len(texts)).tolist()
        split = max(1, round(len(texts) * 0.1))
        validation = [texts[i] for i in order[:split]]
        texts = [texts[i] for i in order[split:]]
    train_data = TextDataset(texts, tokenizer, args.block_size)
    val_data = TextDataset(validation, tokenizer, args.block_size)
    if not len(train_data) or not len(val_data):
        parser.error('Both splits need documents longer than block-size tokens; add data or lower --block-size')
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GPT(tokenizer.get_vocab_size(), args.dim, args.layers, args.heads, args.block_size, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f'Device: {device}; parameters: {sum(p.numel() for p in model.parameters()):,}; train/validation chunks: {len(train_data)}/{len(val_data)}')
    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch}'):
            batch = batch.to(device)
            logits = model(batch[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * batch.size(0)
            count += batch.size(0)
        val_loss = evaluate(model, val_loader, device)
        perplexity = math.exp(val_loss) if val_loss < 700 else float('inf')
        print(f'Epoch {epoch}: train loss={total/count:.4f}; validation loss={val_loss:.4f}; perplexity={perplexity:.2f}')
        if val_loss < best_loss:
            best_loss = val_loss
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'model_state_dict': model.state_dict(), 'config': model.config,
                        'tokenizer_sha256': hashlib.sha256(tokenizer.to_str().encode()).hexdigest(),
                        'epoch': epoch, 'validation_loss': val_loss, 'seed': args.seed}, args.output)
            print(f'Saved best checkpoint: {args.output}')


if __name__ == '__main__':
    main()
