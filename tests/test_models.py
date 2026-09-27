import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from models.deep_models import (
    Vocabulary,
    RNNLanguageModel,
    LSTMLanguageModel,
    GRULanguageModel,
    TransformerLanguageModel,
    predict_next_words
)
from engines.next_word_predictor import NextWordPredictor

class TestDeepModels(unittest.TestCase):
    def setUp(self):
        self.sentences = [
            ["i", "am", "going", "to", "school"],
            ["i", "have", "a", "car"]
        ]
        self.vocab = Vocabulary(min_freq=1)
        self.vocab.build_vocab(self.sentences)

    def test_vocabulary_encoding_decoding(self):
        tokens = ["i", "am", "going"]
        encoded = self.vocab.encode(tokens, add_special_tokens=True)
        self.assertEqual(encoded[0], self.vocab.sos_idx)
        self.assertEqual(encoded[-1], self.vocab.eos_idx)
        decoded = self.vocab.decode(encoded, remove_special_tokens=True)
        self.assertEqual(decoded, tokens)

    def test_model_forward_shapes(self):
        vocab_size = len(self.vocab)
        batch_size = 2
        seq_len = 4
        dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        models = [
            RNNLanguageModel(vocab_size=vocab_size, embed_dim=32, hidden_dim=64, num_layers=1),
            LSTMLanguageModel(vocab_size=vocab_size, embed_dim=32, hidden_dim=64, num_layers=1),
            GRULanguageModel(vocab_size=vocab_size, embed_dim=32, hidden_dim=64, num_layers=1),
            TransformerLanguageModel(vocab_size=vocab_size, embed_dim=32, num_heads=2, hidden_dim=64, num_layers=1)
        ]
        
        for m in models:
            logits, _ = m(dummy_input)
            self.assertEqual(logits.shape, (batch_size, seq_len, vocab_size))

    def test_next_word_predictor(self):
        predictor = NextWordPredictor()
        available = predictor.get_available_models()
        self.assertTrue(len(available) > 0)
        
        preds = predictor.predict("I am going to", model_name="Transformer", top_k=3)
        self.assertIsInstance(preds, list)
        self.assertTrue(len(preds) > 0)
        self.assertIn("word", preds[0])
        self.assertIn("probability", preds[0])

if __name__ == "__main__":
    unittest.main()
