import os
import sys
import time
import json
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.deep_models import (
    Vocabulary,
    RNNLanguageModel,
    LSTMLanguageModel,
    GRULanguageModel,
    TransformerLanguageModel,
    count_parameters,
    predict_next_words
)

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

class TextDataset(Dataset):
    def __init__(self, sequences, vocab):
        self.data = []
        for seq in sequences:
            encoded = vocab.encode(seq, add_special_tokens=True)
            # Create training pairs of length >= 2
            if len(encoded) >= 3:
                for i in range(1, len(encoded)):
                    inp = encoded[:i]
                    tgt = encoded[i]
                    self.data.append((inp, tgt))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch, pad_idx):
    inputs, targets = zip(*batch)
    max_len = max(len(x) for x in inputs)
    padded_inputs = []
    for x in inputs:
        pad_len = max_len - len(x)
        padded_inputs.append(x + [pad_idx] * pad_len)
    return torch.tensor(padded_inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

def train_and_evaluate_model(model, train_loader, val_loader, vocab, epochs=25, lr=0.002, device="cpu"):
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    start_time = time.time()
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            logits, _ = model(inputs)
            # Take the logits at the last valid input position
            last_logits = logits[:, -1, :]
            loss = criterion(last_logits, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item() * targets.size(0)
            total_tokens += targets.size(0)
            
        epoch_train_loss = total_loss / max(total_tokens, 1)
        train_losses.append(epoch_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_tokens = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits, _ = model(inputs)
                last_logits = logits[:, -1, :]
                loss = criterion(last_logits, targets)
                val_loss += loss.item() * targets.size(0)
                val_tokens += targets.size(0)
                
        epoch_val_loss = val_loss / max(val_tokens, 1)
        val_losses.append(epoch_val_loss)

    training_duration = time.time() - start_time
    
    # Calculate Accuracy and Inference Latency on Validation Set
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total_val_samples = 0
    latencies = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            t0 = time.perf_counter()
            logits, _ = model(inputs)
            t1 = time.perf_counter()
            
            latencies.append((t1 - t0) * 1000 / inputs.size(0))  # ms per sample
            
            last_logits = logits[:, -1, :]
            _, top1_preds = torch.max(last_logits, dim=-1)
            _, top5_preds = torch.topk(last_logits, k=min(5, last_logits.size(-1)), dim=-1)
            
            correct_top1 += (top1_preds == targets).sum().item()
            for i in range(targets.size(0)):
                if targets[i] in top5_preds[i]:
                    correct_top5 += 1
            total_val_samples += targets.size(0)

    top1_acc = (correct_top1 / max(total_val_samples, 1)) * 100
    top5_acc = (correct_top5 / max(total_val_samples, 1)) * 100
    avg_latency = sum(latencies) / max(len(latencies), 1)
    
    final_train_loss = train_losses[-1]
    final_val_loss = val_losses[-1]
    perplexity = math.exp(min(final_val_loss, 20.0))
    params = count_parameters(model)
    
    return {
        "train_loss": round(final_train_loss, 4),
        "val_loss": round(final_val_loss, 4),
        "perplexity": round(perplexity, 2),
        "top1_accuracy": round(top1_acc, 2),
        "top5_accuracy": round(top5_acc, 2),
        "inference_latency_ms": round(avg_latency, 2),
        "parameter_count": params,
        "training_time_sec": round(training_duration, 2),
        "train_loss_history": [round(l, 4) for l in train_losses],
        "val_loss_history": [round(l, 4) for l in val_losses]
    }

def main():
    print("=" * 60)
    print("Mini Grammarly - Training RNN, LSTM, GRU & Transformer")
    print("=" * 60)
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    corpus_path = os.path.join(base_dir, "data", "sample_corpus.txt")
    models_dir = os.path.join(base_dir, "models")
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # 1. Load and Tokenize Corpus
    with open(corpus_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
        
    tokenized_sentences = [line.split() for line in lines]
    print(f"Loaded {len(tokenized_sentences)} sentences from corpus.")
    
    # 2. Build Vocabulary
    vocab = Vocabulary(min_freq=1)
    vocab.build_vocab(tokenized_sentences)
    vocab_path = os.path.join(models_dir, "vocab.json")
    vocab.save(vocab_path)
    print(f"Vocabulary built: {len(vocab)} unique tokens (including special tokens). Saved to {vocab_path}")
    
    # 3. Create Dataset and Train/Val split
    random.shuffle(tokenized_sentences)
    split_idx = int(0.85 * len(tokenized_sentences))
    train_sentences = tokenized_sentences[:split_idx]
    val_sentences = tokenized_sentences[split_idx:]
    
    train_dataset = TextDataset(train_sentences, vocab)
    val_dataset = TextDataset(val_sentences, vocab)
    print(f"Training sequences: {len(train_dataset)}, Validation sequences: {len(val_dataset)}")
    
    pad_idx = vocab.pad_idx
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_idx)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_idx)
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")
    
    # 4. Instantiate Models
    models = {
        "RNN": RNNLanguageModel(vocab_size=len(vocab), embed_dim=64, hidden_dim=128, num_layers=2, dropout=0.1).to(device),
        "LSTM": LSTMLanguageModel(vocab_size=len(vocab), embed_dim=64, hidden_dim=128, num_layers=2, dropout=0.1).to(device),
        "GRU": GRULanguageModel(vocab_size=len(vocab), embed_dim=64, hidden_dim=128, num_layers=2, dropout=0.1).to(device),
        "Transformer": TransformerLanguageModel(vocab_size=len(vocab), embed_dim=64, num_heads=4, hidden_dim=128, num_layers=2, dropout=0.1).to(device)
    }
    
    benchmark_results = {}
    test_prompts = [
        "I am going to",
        "I have a",
        "She is going to",
        "Natural language processing is",
        "The students are working"
    ]
    
    for name, model in models.items():
        print(f"\nTraining {name} (Parameters: {count_parameters(model):,})...")
        metrics = train_and_evaluate_model(model, train_loader, val_loader, vocab, epochs=25, lr=0.003, device=device)
        print(f"[{name}] Val Loss: {metrics['val_loss']}, Perplexity: {metrics['perplexity']}, Top-1 Acc: {metrics['top1_accuracy']}%, Latency: {metrics['inference_latency_ms']} ms")
        
        # Save model weights
        weight_path = os.path.join(models_dir, f"{name.lower()}.pt")
        torch.save(model.state_dict(), weight_path)
        print(f"Saved {name} weights to {weight_path}")
        
        # Test sample predictions
        sample_preds = {}
        for prompt in test_prompts:
            preds = predict_next_words(model, vocab, prompt, top_k=3, device=device)
            sample_preds[prompt] = preds
            
        metrics["sample_predictions"] = sample_preds
        benchmark_results[name] = metrics
        
    # Save benchmark results
    results_path = os.path.join(data_dir, "benchmark_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"\nAll models successfully trained! Benchmark results saved to {results_path}")

if __name__ == "__main__":
    main()
