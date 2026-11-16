import argparse
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description='Train byte-level BPE on a UTF-8 corpus')
    parser.add_argument('--files', type=Path, nargs='+', default=[ROOT / 'data/raw/corpus.txt'])
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'tokenizer')
    parser.add_argument('--vocab-size', type=int, default=32000)
    args = parser.parse_args()
    if args.vocab_size < 256:
        parser.error('vocab-size must be at least 256 for byte-level BPE')
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=[str(path) for path in args.files], vocab_size=args.vocab_size, min_frequency=2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_model(str(args.output_dir))
    print(f'Saved tokenizer ({tokenizer.get_vocab_size()} tokens) to {args.output_dir}')


if __name__ == '__main__':
    main()
