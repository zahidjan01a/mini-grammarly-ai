# Mini Grammarly: AI-Powered Writing Assistant ✍️

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14%2BCPU-red.svg)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.64-brightgreen.svg)](https://streamlit.io/)

A complete web-based AI Writing Assistant inspired by Grammarly, featuring deep learning sequence models (**RNN $\rightarrow$ LSTM $\rightarrow$ GRU $\rightarrow$ Transformer**), statistical and contextual spelling correction, syntax and grammar error detection with an interactive suggestion interface, sentence rewriting, readability analytics, voice input, and Urdu-English support.

---

## 🌟 Key Features

### 1. 🔤 Spelling Correction
- Detects spelling errors in sub-milliseconds using **SymSpell** (Symmetric Delete algorithm) with an 82,765-word frequency dictionary.
- Contextual bigram rescoring and phonetic contraction handling (e.g., `goin` $\rightarrow$ `going`, `teh` $\rightarrow$ `the`).
- Non-destructive character span tracking with proposed replacements.

### 2. 📘 Contextual Grammar Correction
- **Subject-Verb Agreement (SVA)**: Identifies number and person mismatches (e.g., `I has a car` $\rightarrow$ `I have a car`, `they is ready` $\rightarrow$ `they are ready`).
- **Auxiliary Verb Inflections**: Enforces correct participle forms (`is walk` $\rightarrow$ `is walking`).
- **Article Usage**: Phonetic vowel-sound enforcement (`an car` $\rightarrow$ `a car`, `a apple` $\rightarrow$ `an apple`).
- **Preposition Appropriateness**: Fixes non-standard preposition usages (`interested on` $\rightarrow$ `interested in`, `good in math` $\rightarrow$ `good at math`).
- **Double Negatives**: Detects and eliminates double negatives (`don't know nothing` $\rightarrow$ `don't know anything`).
- **Comparative Redundancy**: Resolves duplicate comparatives (`more better` $\rightarrow$ `better`).

### 3. 🔮 Deep Next-Word Prediction
- Implements and compares four deep sequence architectures:
  - **Simple RNN**: Baseline recurrent network with sequential hidden state propagation.
  - **LSTM**: Long Short-Term Memory with input, forget, and output gating mechanisms.
  - **GRU**: Gated Recurrent Unit with reset and update gates.
  - **Transformer**: Causal Self-Attention decoder with sinusoidal positional encodings.
- Interactive prediction chips under the text editor: clicking a suggestion automatically appends it to your text!

### 4. ✍️ Sentence Rewriting (Tone Switcher)
- Rephrases text into four distinct stylistic tones:
  - **Formal / Professional** (e.g., replaces slang and contractions with professional phrasing)
  - **Casual / Friendly**
  - **Concise / Direct** (removes wordiness and redundancies)
  - **Academic / Elaborate**

### 5. 📊 Readability & Linguistic Analytics
- Real-time **Flesch Reading Ease** gauge (0–100 scale).
- **Flesch-Kincaid Grade Level** estimation.
- Word count, character count, sentence count, average sentence length, and estimated reading time.

### 6. 💎 Vocabulary Booster
- Highlights overused or basic words and suggests rich alternatives using **NLTK WordNet**.

### 7. 🌐 Roman Urdu to English Support
- Translates Roman Urdu sentences to standard English (e.g., `mein school ja raha hoon` $\rightarrow$ `I am going to school`).

### 8. 🎙️ Browser Voice Input
- Hands-free dictation using the Web Speech API directly into the text editor.

---

## 🚀 Quick Start

### 1. Requirements & Setup
Make sure you have Python 3.10+ installed. Install the dependencies:
```powershell
pip install -r requirements.txt
```
*(Dependencies include `torch`, `transformers`, `streamlit`, `symspellpy`, `nltk`, `textstat`, `pandas`, `altair`)*

### 2. Launch the Application
Run via Streamlit:
```powershell
streamlit run app.py
```
Or double-click the included batch launcher:
```powershell
run_app.bat
```
The app will open automatically in your browser at `http://localhost:8501`.

---

## 🧠 Model Benchmarking & Architecture Comparison

To train all four deep sequence models on the linguistic corpus and generate updated metrics:
```powershell
python scripts/train_models.py
```

### Benchmark Summary

| Architecture | Parameters | Val Loss | Perplexity (PPL) | Top-1 Acc | Top-5 Acc | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Simple RNN** | 125,599 | 8.8068 | 6,679.75 | 8.79% | 21.98% | **0.13 ms** |
| **LSTM** | 299,167 | 9.1656 | 9,562.76 | 2.20% | 14.29% | 0.33 ms |
| **GRU** | 241,311 | 9.1781 | 9,683.19 | 7.69% | 18.68% | 0.31 ms |
| **Transformer** | **112,223** | **8.9096** | **7,402.70** | **8.79%** | **23.08%** | **0.14 ms** |

---

## 🧪 Running the 10 Benchmark Demonstration Sentences

To evaluate all 10 sample sentences across spelling, grammar, next-word predictions, and readability:
```powershell
python scripts/evaluate_examples.py
```

### Example 1 (From PDF Prompt):
- **Input**: `I has a car and I am goin to school.`
- **Detected Spelling**: `goin` $\rightarrow$ `going`
- **Detected Grammar**: `I has` $\rightarrow$ `I have` (Subject-Verb Agreement)
- **Corrected Output**: `I have a car and I am going to school.`

---

## 🔬 Unit Tests
Run the complete automated unit test suite:
```powershell
python -m unittest discover -s tests -p "test_*.py"
```
*(All 10/10 test cases pass with 100% success rate)*

---

## 📁 Repository Structure

```
Mini-Grammarly-Project/
├── app.py                      # Main Streamlit web application
├── run_app.bat                 # Windows 1-click launcher
├── ARCHITECTURE_REPORT.md      # Full academic architecture & comparison report
├── README.md                   # Project overview & documentation
├── data/
│   ├── sample_corpus.txt       # Linguistic training corpus
│   └── benchmark_results.json  # Precomputed benchmark metrics
├── models/
│   ├── deep_models.py          # PyTorch RNN, LSTM, GRU & Transformer architectures
│   ├── vocab.json              # Shared vocabulary mapping
│   ├── rnn.pt                  # Trained Simple RNN weights
│   ├── lstm.pt                 # Trained LSTM weights
│   ├── gru.pt                  # Trained GRU weights
│   └── transformer.pt          # Trained Transformer weights
├── engines/
│   ├── spell_checker.py        # SymSpell & frequency spell checker
│   ├── grammar_checker.py      # Syntactic, SVA & preposition grammar engine
│   ├── next_word_predictor.py  # Multi-model next-word prediction engine
│   └── text_enhancer.py        # Tone rewriter, readability & Urdu support
├── scripts/
│   ├── train_models.py         # Model training & benchmarking script
│   └── evaluate_examples.py    # 10 test sentences evaluation runner
└── tests/
    ├── test_models.py          # Model architecture & forward pass unit tests
    └── test_engines.py         # Spell, grammar & enhancement unit tests
```

---

## 🎓 Academic Statement
This project was developed as an educational prototype to demonstrate how NLP and Deep Learning techniques (RNN, LSTM, GRU, and Self-Attention Transformers) can be combined to build an intelligent, responsive writing assistant.
