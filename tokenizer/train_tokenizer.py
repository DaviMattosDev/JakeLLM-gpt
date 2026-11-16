"""Train and persist a byte-level BPE tokenizer using user-supplied local texts."""

import argparse
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train JakeLLM's byte-level BPE tokenizer")
    parser.add_argument("files", nargs="*", default=["data/raw/corpus.txt"])
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--output-dir", default="tokenizer")
    args = parser.parse_args(argv)
    for file in args.files:
        if not Path(file).is_file():
            parser.error(f"Corpus does not exist: {file}")
    if args.vocab_size < 260:
        parser.error("--vocab-size must be at least 260 for byte-level BPE plus special tokens")
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=args.files, vocab_size=args.vocab_size, min_frequency=args.min_frequency,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
    )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    tokenizer.save_model(args.output_dir)
    print(f"Saved tokenizer ({tokenizer.get_vocab_size()} tokens) in {args.output_dir}")


if __name__ == "__main__":
    main()
