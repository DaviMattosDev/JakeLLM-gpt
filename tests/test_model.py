import pytest
import torch

from model.gpt_model import GPT
from model.train import TextDataset


def tiny_model(block_size=8):
    torch.manual_seed(123)
    return GPT(vocab_size=32, dim=16, layers=2, heads=4, block_size=block_size).eval()


def test_logits_have_expected_shape():
    model = tiny_model()
    assert model(torch.tensor([[1, 2, 3], [4, 5, 6]])).shape == (2, 3, 32)


def test_causal_mask_blocks_information_from_future():
    model = tiny_model()
    original = torch.tensor([[1, 2, 3, 4, 5]])
    edited = torch.tensor([[1, 2, 3, 4, 6]])
    with torch.inference_mode():
        a = model(original)
        b = model(edited)
    torch.testing.assert_close(a[:, :4], b[:, :4], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(a[:, -1], b[:, -1])


def test_sequence_length_is_checked():
    model = tiny_model(block_size=4)
    with pytest.raises(ValueError, match="Sequence length"):
        model(torch.ones((1, 5), dtype=torch.long))
    with pytest.raises(ValueError, match="Sequence length"):
        model(torch.ones((1, 0), dtype=torch.long))


def test_generation_truncates_context_and_can_be_greedy():
    model = tiny_model(block_size=4)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    out_a = model.generate(ids, max_new_tokens=5, temperature=0)
    out_b = model.generate(ids, max_new_tokens=5, temperature=0)
    assert out_a.shape == (1, 11)
    assert torch.equal(out_a, out_b)
    assert torch.equal(out_a[:, :6], ids)


def test_generate_rejects_invalid_inputs():
    model = tiny_model()
    with pytest.raises(ValueError, match="nonnegative"):
        model.generate(torch.tensor([[1]]), temperature=-1)
    with pytest.raises(ValueError, match="nonempty"):
        model.generate(torch.empty((1, 0), dtype=torch.long))


def test_dataset_includes_exact_full_window():
    data = TextDataset(list(range(129)), block_size=128)
    assert len(data) == 1
    x, y = data[0]
    assert x.tolist() == list(range(128))
    assert y.tolist() == list(range(1, 129))


def test_dataset_drops_incomplete_windows():
    assert len(TextDataset(list(range(128)), block_size=128)) == 0
    assert len(TextDataset(list(range(130)), block_size=128)) == 1
