import os
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

class Vocabulary:
    def __init__(self, min_freq=1):
        self.min_freq = min_freq
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.sos_token = "<sos>"
        self.eos_token = "<eos>"
        
        self.special_tokens = [self.pad_token, self.unk_token, self.sos_token, self.eos_token]
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = {}
        
        for idx, token in enumerate(self.special_tokens):
            self.word2idx[token] = idx
            self.idx2word[idx] = token

    @property
    def pad_idx(self):
        return self.word2idx[self.pad_token]

    @property
    def unk_idx(self):
        return self.word2idx[self.unk_token]

    @property
    def sos_idx(self):
        return self.word2idx[self.sos_token]

    @property
    def eos_idx(self):
        return self.word2idx[self.eos_token]

    def build_vocab(self, tokenized_texts):
        for tokens in tokenized_texts:
            for token in tokens:
                token = token.lower()
                self.word_counts[token] = self.word_counts.get(token, 0) + 1
        
        for token, count in self.word_counts.items():
            if count >= self.min_freq and token not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[token] = idx
                self.idx2word[idx] = token

    def __len__(self):
        return len(self.word2idx)

    def encode(self, tokens, add_special_tokens=True):
        indices = []
        if add_special_tokens:
            indices.append(self.sos_idx)
        for token in tokens:
            token = token.lower()
            indices.append(self.word2idx.get(token, self.unk_idx))
        if add_special_tokens:
            indices.append(self.eos_idx)
        return indices

    def decode(self, indices, remove_special_tokens=True):
        words = []
        for idx in indices:
            if idx in self.idx2word:
                word = self.idx2word[idx]
                if remove_special_tokens and word in self.special_tokens:
                    continue
                words.append(word)
        return words

    def save(self, filepath):
        data = {
            "word2idx": self.word2idx,
            "min_freq": self.min_freq
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls(min_freq=data.get("min_freq", 1))
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = {int(v): k for k, v in vocab.word2idx.items()}
        return vocab


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.2):
        super().__init__()
        self.model_type = "RNN"
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embeds = self.dropout(self.embedding(x))
        out, hidden = self.rnn(embeds, hidden)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits, hidden


class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.2):
        super().__init__()
        self.model_type = "LSTM"
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embeds = self.dropout(self.embedding(x))
        out, hidden = self.lstm(embeds, hidden)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits, hidden


class GRULanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.2):
        super().__init__()
        self.model_type = "GRU"
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embeds = self.dropout(self.embedding(x))
        out, hidden = self.gru(embeds, hidden)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits, hidden


class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, hidden_dim=256, num_layers=2, dropout=0.2, max_len=128):
        super().__init__()
        self.model_type = "Transformer"
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x, hidden=None):
        seq_len = x.size(1)
        mask = self.generate_square_subsequent_mask(seq_len, x.device)
        
        embeds = self.embedding(x) * math.sqrt(self.embed_dim)
        embeds = self.pos_encoder(embeds)
        embeds = self.dropout(embeds)
        
        out = self.transformer_encoder(embeds, mask=mask, is_causal=True)
        logits = self.fc(out)
        return logits, None


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def predict_next_words(model, vocab, text_prefix, top_k=5, temperature=1.0, device="cpu"):
    """
    Given a model and prefix text, compute probabilities for the next token and return Top-K words.
    """
    model.eval()
    tokens = text_prefix.strip().split()
    if not tokens:
        tokens = [vocab.sos_token]
    
    encoded = vocab.encode(tokens, add_special_tokens=True)
    # Remove EOS token for prediction prompt
    if encoded[-1] == vocab.eos_idx:
        encoded = encoded[:-1]
        
    input_tensor = torch.tensor([encoded], dtype=torch.long).to(device)
    
    with torch.no_grad():
        logits, _ = model(input_tensor)
        next_token_logits = logits[0, -1, :] / max(temperature, 1e-4)
        
        # Suppress special tokens from being suggested as normal words
        for st in [vocab.pad_idx, vocab.unk_idx, vocab.sos_idx]:
            next_token_logits[st] = -1e9
            
        probs = F.softmax(next_token_logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, k=min(top_k, len(vocab)))
        
        suggestions = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            word = vocab.idx2word.get(idx, "<unk>")
            if word not in [vocab.pad_token, vocab.unk_token, vocab.sos_token, vocab.eos_token]:
                suggestions.append({"word": word, "probability": round(prob, 4), "percentage": f"{prob*100:.1f}%"})
                
        return suggestions
