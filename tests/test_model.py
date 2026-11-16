import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import ByteLevelBPETokenizer

from inference.cli import generate, load_model
from model.gpt_model import GPT
from model.train import TextDataset


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        self.tokenizer = ByteLevelBPETokenizer()
        self.tokenizer.train_from_iterator(['O cachorro Jake gosta de brincar. ' * 20], vocab_size=300)
        self.model = GPT(self.tokenizer.get_vocab_size(), dim=16, layers=1, heads=2, block_size=8, dropout=0)

    def test_future_tokens_cannot_change_prefix_predictions(self):
        self.model.eval()
        left = torch.tensor([[1, 2, 3, 4, 5]])
        right = torch.tensor([[1, 2, 3, 9, 10]])
        with torch.no_grad():
            torch.testing.assert_close(self.model(left)[:, :3], self.model(right)[:, :3])

    def test_can_learn_a_small_batch(self):
        batch = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.02)
        losses = []
        for _ in range(25):
            logits = self.model(batch)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch.roll(-1, 1).reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], losses[0] * 0.5)

    def test_chunk_has_extra_target_token(self):
        data = TextDataset(['O cachorro Jake gosta de brincar. ' * 5], self.tokenizer, block_size=8)
        self.assertEqual(data[0].numel(), 9)
        torch.testing.assert_close(data[0], torch.tensor(self.tokenizer.encode('O cachorro Jake gosta de brincar. ' * 5).ids[:9]))

    def test_generation_beyond_context_and_invalid_inputs(self):
        result = generate(self.model, self.tokenizer, 'O cachorro Jake gosta de brincar. ' * 5, max_len=12)
        self.assertTrue(result)
        for kwargs in [dict(prompt=''), dict(temperature=0), dict(temperature=float('nan')), dict(top_k=0), dict(max_len=-1)]:
            with self.assertRaises(ValueError):
                generate(self.model, self.tokenizer, **({'prompt': 'Jake'} | kwargs))

    def test_checkpoint_roundtrip_and_tokenizer_mismatch(self):
        import hashlib
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'weights.pth'
            checkpoint = dict(model_state_dict=self.model.state_dict(), config=self.model.config,
                              tokenizer_sha256=hashlib.sha256(self.tokenizer.to_str().encode()).hexdigest())
            torch.save(checkpoint, path)
            restored = load_model(path, self.tokenizer)
            sample = torch.tensor([[1, 2, 3]])
            self.model.eval()
            torch.testing.assert_close(self.model(sample), restored(sample))
            checkpoint['tokenizer_sha256'] = 'wrong'
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, 'Tokenizer differs'):
                load_model(path, self.tokenizer)

    def test_context_limit_has_clear_error(self):
        with self.assertRaises(ValueError):
            self.model(torch.ones(1, 9, dtype=torch.long))


if __name__ == '__main__':
    unittest.main()
