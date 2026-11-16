"""Artifact loading and text generation shared by CLI and HTTP API."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from model.gpt_model import GPT

if TYPE_CHECKING:
    from tokenizers import ByteLevelBPETokenizer


def load_model(
    checkpoint_path: str = "checkpoints/jakellm.pt",
    tokenizer_dir: str = "tokenizer",
    device: str = "auto",
):
    tokenizer_path = Path(tokenizer_dir)
    vocab, merges = tokenizer_path / "vocab.json", tokenizer_path / "merges.txt"
    if not vocab.is_file() or not merges.is_file():
        raise FileNotFoundError("Tokenizer files missing. Run python -m tokenizer.train_tokenizer")
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"Checkpoint missing: {checkpoint_path}. Run python -m model.train")
    from tokenizers import ByteLevelBPETokenizer

    tokenizer = ByteLevelBPETokenizer.from_file(str(vocab), str(merges))
    resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else (
        "cpu" if device == "auto" else device
    )
    # Only trusted local checkpoints should be loaded; weights_only blocks arbitrary pickles.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 2:
        raise ValueError("Legacy checkpoint is incompatible with the causal model; retrain JakeLLM")
    config = checkpoint["model_config"]
    if config["vocab_size"] != tokenizer.get_vocab_size():
        raise ValueError("Tokenizer vocabulary does not match the checkpoint")
    model = GPT(**config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(resolved_device).eval()
    return model, tokenizer


def generate_text(
    model: GPT,
    tokenizer: ByteLevelBPETokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 20,
) -> str:
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty")
    ids = tokenizer.encode(prompt).ids
    if not ids:
        raise ValueError("Tokenizer returned no tokens")
    input_ids = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)
    eos_token_id = tokenizer.token_to_id("<eos>")
    result_ids = model.generate(
        input_ids, max_new_tokens=max_new_tokens, temperature=temperature,
        top_k=top_k, eos_token_id=eos_token_id,
    )[0].tolist()
    generated = result_ids[len(ids):]
    if eos_token_id is not None and generated and generated[-1] == eos_token_id:
        generated = generated[:-1]
    return tokenizer.decode(generated, skip_special_tokens=True)
