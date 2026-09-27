import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engines.spell_checker import SpellCheckerEngine
from engines.grammar_checker import GrammarCheckerEngine
from engines.next_word_predictor import NextWordPredictor
from engines.text_enhancer import TextEnhancerEngine

def main():
    spell_eng = SpellCheckerEngine()
    grammar_eng = GrammarCheckerEngine()
    next_word_eng = NextWordPredictor()
    enhancer_eng = TextEnhancerEngine()

    test_sentences = [
        ("PDF Example 1", "I has a car and I am goin to school."),
        ("PDF Example 2", "I am going to"),
        ("Subject-Verb Agreement", "He have a new laptop and they is studying in library."),
        ("Spelling & Slang", "They are talkin and playin in teh garden."),
        ("Article Misuse", "He is reading a interesting book about an machine learning."),
        ("Preposition Error", "She is very interested on artificial intelligence."),
        ("Verb Tense with 'Did'", "We did went to the university yesterday."),
        ("Double Negative", "The students don't know nothing about the exam schedule."),
        ("Comparative Redundancy", "This transformer model is more better and more faster."),
        ("Roman Urdu Support", "mein school ja raha hoon")
    ]

    print("=" * 80)
    print("MINI GRAMMARLY - 10 DEMONSTRATIVE BENCHMARK EVALUATIONS")
    print("=" * 80)

    results = []

    for name, text in test_sentences:
        print(f"\n--- [{name}] ---")
        print(f"Original Text: '{text}'")

        if name == "Roman Urdu Support":
            translated = enhancer_eng.translate_urdu(text)
            print(f"Urdu Translation: '{translated}'")
            text = translated

        # Check spelling
        spell_errs = spell_eng.check(text)
        spell_str = ", ".join([f"{e['original']} -> {e['replacement']}" for e in spell_errs]) if spell_errs else "None"
        print(f"Spelling Issues: {spell_str}")

        # Check grammar
        gram_errs = grammar_eng.check(text)
        gram_str = ", ".join([f"{e['original']} -> {e['replacement']} ({e.get('category', 'Grammar')})" for e in gram_errs]) if gram_errs else "None"
        print(f"Grammar Issues: {gram_str}")

        # Auto-correct
        corrected = grammar_eng.auto_correct(spell_eng.auto_correct(text))
        print(f"Corrected Sentence: '{corrected}'")

        # Next-word predictions
        preds = next_word_eng.predict(corrected, model_name="Transformer", top_k=3)
        pred_str = ", ".join([f"[{p['word']}: {p['percentage']}]" for p in preds]) if preds else "N/A"
        print(f"Next-Word Predictions (Transformer): {pred_str}")

        # Readability
        read_stats = enhancer_eng.analyze_readability(corrected)
        print(f"Readability: Flesch Ease {read_stats['reading_ease']} ({read_stats['reading_ease_label']}), Grade {read_stats['grade_level']}")

        results.append({
            "test_case": name,
            "original": text,
            "spelling_errors": spell_str,
            "grammar_errors": gram_str,
            "corrected": corrected,
            "next_words": pred_str,
            "readability": f"Ease {read_stats['reading_ease']} | Grade {read_stats['grade_level']}"
        })

    print("\n" + "=" * 80)
    print("Evaluation completed successfully.")

if __name__ == "__main__":
    main()
