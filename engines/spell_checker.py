import os
import re
import symspellpy
from symspellpy import SymSpell, Verbosity

class SpellCheckerEngine:
    def __init__(self, max_edit_distance=2, prefix_length=7):
        self.sym_spell = SymSpell(max_dictionary_edit_distance=max_edit_distance, prefix_length=prefix_length)
        base_sym_dir = os.path.dirname(symspellpy.__file__)
        dictionary_path = os.path.join(base_sym_dir, "frequency_dictionary_en_82_765.txt")
        self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        
        bigram_path = os.path.join(base_sym_dir, "frequency_bigramdictionary_en_243_342.txt")
        if os.path.exists(bigram_path):
            try:
                self.sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)
            except Exception:
                pass

        # Domain words to whitelist (Mini Grammarly / NLP specific terms)
        custom_words = [
            "grammarly", "rnn", "lstm", "gru", "transformer", "nlp", "ai",
            "pytorch", "tensorflow", "tokenization", "seq2seq", "urdu",
            "flesch", "kincaid", "dataset", "pretrained", "backpropagation",
            "recurrent", "decoder", "self-attention", "subagent", "streamlit"
        ]
        for word in custom_words:
            self.sym_spell.create_dictionary_entry(word, 1000000)

        # High-frequency phonetic & colloquial typos
        self.common_typos = {
            "goin": "going",
            "doin": "doing",
            "workin": "working",
            "playin": "playing",
            "sleepin": "sleeping",
            "eatin": "eating",
            "writin": "writing",
            "teh": "the",
            "recieved": "received",
            "recieve": "receive",
            "seperate": "separate",
            "untill": "until",
            "definately": "definitely",
            "accommodate": "accommodate",
            "becuase": "because",
            "beleive": "believe",
            "goverment": "government",
            "tommorrow": "tomorrow",
            "truely": "truly"
        }

    def check(self, text):
        """
        Analyze text for spelling errors.
        Returns list of structured error objects with character offsets and suggestions.
        """
        if not text or not text.strip():
            return []

        errors = []
        # Find all word tokens with their start and end character positions
        word_matches = list(re.finditer(r"\b[A-Za-z'-]+\b", text))

        for idx, match in enumerate(word_matches):
            raw_word = match.group(0)
            clean_word = raw_word.lower()
            start, end = match.span()

            # Skip words with apostrophes that are contractions or possessives (e.g., don't, user's)
            if "'" in clean_word:
                base = clean_word.split("'")[0]
                if self.sym_spell.words.get(base, 0) > 0 or self.sym_spell.words.get(clean_word, 0) > 0:
                    continue

            # Skip single letters except 'a' and 'i'
            if len(clean_word) == 1 and clean_word not in ['a', 'i']:
                continue

            # Check if word exists in dictionary
            freq = self.sym_spell.words.get(clean_word, 0)
            if freq > 0:
                continue  # Correct word

            # Previous and next word for context
            prev_word = word_matches[idx - 1].group(0).lower() if idx > 0 else ""
            next_word = word_matches[idx + 1].group(0).lower() if idx + 1 < len(word_matches) else ""

            # Check direct high-confidence typo dictionary
            if clean_word in self.common_typos:
                direct_suggestion = self.common_typos[clean_word]
                candidate_terms = [direct_suggestion]
            elif clean_word.endswith("in") and (clean_word + "g") in self.sym_spell.words:
                candidate_terms = [clean_word + "g"]
            else:
                candidate_terms = []

            # Look up suggestions via SymSpell
            suggestions_lookup = self.sym_spell.lookup(
                clean_word,
                Verbosity.CLOSEST,
                max_edit_distance=2,
                include_unknown=False
            )

            # Score candidates with bigram context if available
            scored_candidates = []
            for s in suggestions_lookup:
                term = s.term
                score = s.count
                if prev_word:
                    bigram = f"{prev_word} {term}"
                    if hasattr(self.sym_spell, "bigrams") and bigram in self.sym_spell.bigrams:
                        score += self.sym_spell.bigrams[bigram] * 10
                if next_word:
                    bigram = f"{term} {next_word}"
                    if hasattr(self.sym_spell, "bigrams") and bigram in self.sym_spell.bigrams:
                        score += self.sym_spell.bigrams[bigram] * 10
                scored_candidates.append((term, score))

            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            for term, _ in scored_candidates:
                if term not in candidate_terms:
                    candidate_terms.append(term)

            if candidate_terms:
                formatted_suggestions = []
                for term in candidate_terms[:4]:
                    if raw_word.isupper():
                        term = term.upper()
                    elif raw_word[0].isupper():
                        term = term.capitalize()
                    formatted_suggestions.append(term)

                best_suggestion = formatted_suggestions[0]

                errors.append({
                    "id": f"spell_{idx}_{start}",
                    "type": "Spelling",
                    "severity": "critical",
                    "original": raw_word,
                    "replacement": best_suggestion,
                    "suggestions": formatted_suggestions,
                    "start": start,
                    "end": end,
                    "message": f"Possible spelling error: '{raw_word}'. Suggested correction: '{best_suggestion}'.",
                    "explanation": f"The word '{raw_word}' is not recognized in the dictionary. Did you mean '{best_suggestion}'?"
                })

        return errors

    def auto_correct(self, text):
        """
        Auto-corrects all detected spelling errors in text.
        """
        errors = self.check(text)
        if not errors:
            return text

        # Replace in reverse order of offsets to keep indices intact
        chars = list(text)
        for err in sorted(errors, key=lambda x: x["start"], reverse=True):
            chars[err["start"]:err["end"]] = list(err["replacement"])

        return "".join(chars)
