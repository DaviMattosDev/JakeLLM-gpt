from pathlib import Path

import pytest
import torch

pytest.importorskip("tokenizers")
from tokenizers import ByteLevelBPETokenizer

from inference.generate import generate_text, load_model
from model.gpt_model import GPT


def test_checkpoint_load_and_infer(tmp_path):
    tok = ByteLevelBPETokenizer()
    tok.train_from_iterator(["Olá Jake, tudo bem? " * 20], vocab_size=300,
                            special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"])
    tok_dir = tmp_path / "tok"
    tok_dir.mkdir()
    tok.save_model(str(tok_dir))
    model = GPT(vocab_size=tok.get_vocab_size(), dim=16, layers=1, heads=4, block_size=8)
    ckpt = tmp_path / "model.pt"
    torch.save({"format_version": 2, "model_config": model.config,
                "model_state_dict": model.state_dict()}, ckpt)
    restored, tokenizer = load_model(str(ckpt), str(tok_dir), "cpu")
    result = generate_text(restored, tokenizer, "Olá", max_new_tokens=3, temperature=0)
    assert isinstance(result, str)


def test_legacy_checkpoints_are_rejected(tmp_path):
    tok = ByteLevelBPETokenizer()
    tok.train_from_iterator(["Olá Jake " * 15], vocab_size=300)
    tok_dir = tmp_path / "tok"
    tok_dir.mkdir()
    tok.save_model(str(tok_dir))
    ckpt = tmp_path / "legacy.pt"
    torch.save({"some_old_weight": torch.tensor([1])}, ckpt)
    with pytest.raises(ValueError, match="Legacy checkpoint"):
        load_model(str(ckpt), str(tok_dir), "cpu")
