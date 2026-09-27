import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from engines.spell_checker import SpellCheckerEngine
from engines.grammar_checker import GrammarCheckerEngine
from engines.text_enhancer import TextEnhancerEngine

class TestEngines(unittest.TestCase):
    def setUp(self):
        self.spell_engine = SpellCheckerEngine()
        self.grammar_engine = GrammarCheckerEngine()
        self.enhancer_engine = TextEnhancerEngine()

    def test_spelling_detection_and_autocorrect(self):
        text = "I am goin to school."
        errors = self.spell_engine.check(text)
        self.assertTrue(len(errors) > 0)
        self.assertEqual(errors[0]["original"], "goin")
        self.assertEqual(errors[0]["replacement"], "going")
        
        corrected = self.spell_engine.auto_correct(text)
        self.assertIn("going", corrected)

    def test_grammar_subject_verb_agreement(self):
        # Specific PDF Example: "I has a car and I am goin to school."
        text = "I has a car and I am goin to school."
        grammar_errors = self.grammar_engine.check(text)
        self.assertTrue(len(grammar_errors) > 0)
        
        # Check that 'I has' is flagged and suggested replacement is 'I have'
        sva_error = next((e for e in grammar_errors if "I has" in e["original"]), None)
        self.assertIsNotNone(sva_error)
        self.assertIn("I have", sva_error["replacement"])

    def test_article_check(self):
        text = "He bought a apple and an car."
        errors = self.grammar_engine.check(text)
        self.assertTrue(len(errors) >= 2)
        replacements = [e["replacement"] for e in errors]
        self.assertTrue(any("an apple" in r.lower() for r in replacements))
        self.assertTrue(any("a car" in r.lower() for r in replacements))

    def test_preposition_check(self):
        text = "She is interested on deep learning."
        errors = self.grammar_engine.check(text)
        self.assertTrue(len(errors) > 0)
        self.assertTrue(any("interested in" in e["replacement"] for e in errors))

    def test_readability_analysis(self):
        text = "Natural language processing is a branch of artificial intelligence that helps computers understand human language."
        metrics = self.enhancer_engine.analyze_readability(text)
        self.assertGreater(metrics["word_count"], 5)
        self.assertIn("reading_ease", metrics)
        self.assertIn("grade_level", metrics)

    def test_sentence_rewriting(self):
        text = "I think kids gonna make big problems."
        formal = self.enhancer_engine.rewrite_sentence(text, tone="Formal")
        self.assertNotIn("gonna", formal)
        self.assertIn("children", formal)

    def test_urdu_translation(self):
        urdu_text = "mein school ja raha hoon"
        eng = self.enhancer_engine.translate_urdu(urdu_text)
        self.assertIn("school", eng.lower())
        self.assertIn("going", eng.lower())

if __name__ == "__main__":
    unittest.main()
