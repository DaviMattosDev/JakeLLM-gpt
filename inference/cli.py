"""Interactive generation: python -m inference.cli --help."""
import argparse
import hashlib
import math
from pathlib import Path

import torch
from tokenizers import ByteLevelBPETokenizer

from model.gpt_model import GPT

ROOT = Path(__file__).resolve().parents[1]


def load_model(checkpoint_path, tokenizer, device='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if 'model_state_dict' in checkpoint:
        fingerprint = hashlib.sha256(tokenizer.to_str().encode()).hexdigest()
        if checkpoint.get('tokenizer_sha256', fingerprint) != fingerprint:
            raise ValueError('Tokenizer differs from the one used to train this checkpoint')
        model = GPT(**checkpoint['config'])
        state = checkpoint['model_state_dict']
    else:
        # Original repository checkpoints used these architecture defaults.
        model = GPT(vocab_size=tokenizer.get_vocab_size())
        state = checkpoint
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def generate(model, tokenizer, prompt, max_len=50, temperature=1.0, top_k=20):
    if not isinstance(max_len, int) or max_len < 0:
        raise ValueError('max_len must be a nonnegative integer')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and positive')
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError('top_k must be a positive integer')
    tokens = tokenizer.encode(prompt).ids
    if not tokens:
        raise ValueError('Enter a nonempty prompt')
    model.eval()
    device = next(model.parameters()).device
    for _ in range(max_len):
        context = torch.tensor([tokens[-model.block_size:]], dtype=torch.long, device=device)
        logits = model(context)[0, -1] / temperature
        values, indices = torch.topk(logits, min(top_k, logits.size(-1)))
        next_token = indices[torch.multinomial(values.softmax(dim=-1), 1)].item()
        tokens.append(next_token)
    return tokenizer.decode(tokens)


def main():
    parser = argparse.ArgumentParser(description='Converse com o JakeLLM')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'model_weights.pth')
    parser.add_argument('--tokenizer-dir', type=Path, default=ROOT / 'tokenizer')
    parser.add_argument('--max-new-tokens', type=int, default=50)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=20)
    args = parser.parse_args()
    tokenizer = ByteLevelBPETokenizer(str(args.tokenizer_dir / 'vocab.json'), str(args.tokenizer_dir / 'merges.txt'))
    model = load_model(args.checkpoint, tokenizer)
    print("JakeLLM pronto! Digite 'sair' para encerrar.")
    while True:
        try:
            prompt = input('Você: ')
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.lower() in ['sair', 'exit', 'quit']:
            break
        try:
            print('JakeLLM:', generate(model, tokenizer, prompt, args.max_new_tokens, args.temperature, args.top_k))
        except ValueError as error:
            print(error)


if __name__ == '__main__':
    main()
