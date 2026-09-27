import os
import sys
import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.deep_models import (
    Vocabulary,
    RNNLanguageModel,
    LSTMLanguageModel,
    GRULanguageModel,
    TransformerLanguageModel,
    predict_next_words
)

class NextWordPredictor:
    def __init__(self, models_dir=None):
        if models_dir is None:
            models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
        self.models_dir = models_dir
        self.vocab = None
        self.models = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_vocab()
        self._load_available_models()

    def _load_vocab(self):
        vocab_path = os.path.join(self.models_dir, "vocab.json")
        if os.path.exists(vocab_path):
            self.vocab = Vocabulary.load(vocab_path)

    def _load_available_models(self):
        if not self.vocab:
            return

        vocab_size = len(self.vocab)
        model_classes = {
            "RNN": lambda: RNNLanguageModel(vocab_size=vocab_size, embed_dim=64, hidden_dim=128, num_layers=2, dropout=0.1),
            "LSTM": lambda: LSTMLanguageModel(vocab_size=vocab_size, embed_dim=64, hidden_dim=128, num_layers=2, dropout=0.1),
            "GRU": lambda: GRULanguageModel(vocab_size=vocab_size, embed_dim=64, hidden_dim=128, num_layers=2, dropout=0.1),
            "Transformer": lambda: TransformerLanguageModel(vocab_size=vocab_size, embed_dim=64, num_heads=4, hidden_dim=128, num_layers=2, dropout=0.1)
        }

        for name, constructor in model_classes.items():
            weight_path = os.path.join(self.models_dir, f"{name.lower()}.pt")
            if os.path.exists(weight_path):
                try:
                    model = constructor().to(self.device)
                    model.load_state_dict(torch.load(weight_path, map_location=self.device))
                    model.eval()
                    self.models[name] = model
                except Exception as e:
                    print(f"Error loading {name}: {e}")

    def get_available_models(self):
        # Always make sure models are up to date if trained recently
        if not self.vocab:
            self._load_vocab()
        if len(self.models) < 4:
            self._load_available_models()
        return list(self.models.keys())

    def predict(self, text, model_name="Transformer", top_k=5, temperature=1.0):
        """
        Predict next words using the specified model.
        """
        if not self.vocab or not self.models:
            self._load_vocab()
            self._load_available_models()

        if not self.models:
            # Fallback heuristic if models have not finished training yet
            return self._heuristic_fallback(text, top_k)

        if model_name not in self.models:
            # Fallback to first available model
            model_name = next(iter(self.models))

        model = self.models[model_name]
        return predict_next_words(model, self.vocab, text, top_k=top_k, temperature=temperature, device=self.device)

    def compare_all_models(self, text, top_k=3):
        """
        Run inference across all available models for side-by-side comparison.
        """
        results = {}
        for name in self.get_available_models():
            results[name] = self.predict(text, model_name=name, top_k=top_k)
        return results

    def _heuristic_fallback(self, text, top_k=5):
        """
        Graceful fallback when model weights are initializing.
        """
        tokens = text.lower().strip().split()
        last_word = tokens[-1] if tokens else ""
        
        common_followups = {
            "to": ["school", "market", "university", "work", "buy", "learn"],
            "going": ["to", "home", "back", "there"],
            "am": ["going", "working", "studying", "reading", "happy"],
            "have": ["a", "been", "finished", "to", "many"],
            "has": ["a", "finished", "been", "become"],
            "a": ["car", "book", "computer", "doctor", "student"],
            "is": ["going", "an", "a", "important", "essential"],
            "the": ["students", "professor", "team", "algorithm", "world"]
        }
        
        words = common_followups.get(last_word, ["to", "and", "is", "for", "the", "in", "with"])
        suggestions = []
        for i, w in enumerate(words[:top_k]):
            prob = round(1.0 / (i + 1.5), 3)
            suggestions.append({"word": w, "probability": prob, "percentage": f"{prob*100:.1f}%"})
        return suggestions
