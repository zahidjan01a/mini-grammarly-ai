import re
import nltk
from nltk.corpus import wordnet
import textstat

class TextEnhancerEngine:
    def __init__(self):
        # Basic or overused words mapping to advanced vocabulary suggestions
        self.vocab_upgrades = {
            "good": ["excellent", "superb", "commendable", "exceptional", "favorable"],
            "bad": ["adverse", "substandard", "detrimental", "unfavorable", "flawed"],
            "big": ["substantial", "immense", "significant", "prominent", "extensive"],
            "small": ["compact", "diminutive", "marginal", "negligible", "concise"],
            "happy": ["delighted", "content", "gratified", "thrilled", "jubilant"],
            "said": ["stated", "remarked", "articulated", "asserted", "conveyed"],
            "important": ["crucial", "paramount", "imperative", "pivotal", "indispensable"],
            "help": ["assist", "facilitate", "support", "reinforce", "collaborate"],
            "think": ["contemplate", "deliberate", "conceive", "surmise", "postulate"],
            "make": ["generate", "fabricate", "formulate", "construct", "produce"],
            "problem": ["dilemma", "complication", "challenge", "obstacle", "impediment"],
            "show": ["demonstrate", "illustrate", "exhibit", "manifest", "depict"],
            "use": ["utilize", "deploy", "employ", "leverage", "implement"],
            "get": ["acquire", "obtain", "derive", "secure", "procure"],
            "very": ["exceptionally", "profoundly", "exceedingly", "remarkably", "substantially"],
            "hard": ["arduous", "demanding", "rigorous", "intricate", "formidable"],
            "fast": ["rapid", "swift", "expeditious", "prompt", "accelerated"]
        }

        # Roman Urdu common conversational phrases translation dictionary
        self.urdu_dictionary = {
            "mein school ja raha hoon": "I am going to school",
            "main school ja raha hoon": "I am going to school",
            "mere paas gaari hai": "I have a car",
            "mera naam": "my name is",
            "yeh bohat acha hai": "this is very good",
            "yeh bohot acha hai": "this is very good",
            "mujhe madad chahiye": "I need help",
            "woh university ja rahi hai": "she is going to university",
            "hum market ja rahe hain": "we are going to market",
            "aaj mausam kaisa hai": "how is the weather today",
            "shukriya": "thank you",
            "khushamdeed": "welcome",
            "allah hafiz": "goodbye",
            "kya haal hai": "how are you",
            "theek hoon": "I am fine"
        }

    def analyze_readability(self, text):
        """
        Compute comprehensive readability scores and text statistics.
        """
        if not text or not text.strip():
            return {
                "word_count": 0,
                "character_count": 0,
                "sentence_count": 0,
                "reading_ease": 0,
                "grade_level": 0,
                "reading_ease_label": "N/A",
                "avg_sentence_len": 0,
                "reading_time_sec": 0
            }

        word_count = textstat.lexicon_count(text, removepunct=True)
        char_count = len(text)
        sentence_count = max(textstat.sentence_count(text), 1)
        reading_ease = round(textstat.flesch_reading_ease(text), 1)
        grade_level = round(textstat.flesch_kincaid_grade(text), 1)
        avg_sentence_len = round(word_count / sentence_count, 1)
        # Average adult reading speed: 200 words per minute (~3.3 words per second)
        reading_time_sec = round((word_count / 200) * 60, 1)

        # Interpret Flesch Reading Ease
        if reading_ease >= 80:
            ease_label = "Very Easy to Read"
        elif reading_ease >= 65:
            ease_label = "Standard / Plain English"
        elif reading_ease >= 50:
            ease_label = "Fairly Difficult"
        elif reading_ease >= 30:
            ease_label = "Difficult (College Level)"
        else:
            ease_label = "Very Confusing / Technical"

        return {
            "word_count": word_count,
            "character_count": char_count,
            "sentence_count": sentence_count,
            "reading_ease": reading_ease,
            "grade_level": grade_level,
            "reading_ease_label": ease_label,
            "avg_sentence_len": avg_sentence_len,
            "reading_time_sec": reading_time_sec
        }

    def get_synonym_suggestions(self, text):
        """
        Identify overused or simple words and provide advanced vocabulary suggestions.
        """
        words = re.findall(r"\b[A-Za-z]+\b", text)
        suggestions = []

        seen_words = set()
        for w in words:
            w_lower = w.lower()
            if w_lower in seen_words:
                continue
            seen_words.add(w_lower)

            # Check curated upgrade list
            if w_lower in self.vocab_upgrades:
                suggestions.append({
                    "original": w,
                    "upgrades": self.vocab_upgrades[w_lower],
                    "category": "Vocabulary Enhancement"
                })
            else:
                # Fallback to WordNet synonyms for rich alternatives
                synsets = wordnet.synsets(w_lower)
                lemmas = set()
                for s in synsets[:2]:
                    for l in s.lemmas():
                        clean_lemma = l.name().replace("_", " ").lower()
                        if clean_lemma != w_lower and len(clean_lemma) > 2:
                            lemmas.add(clean_lemma)
                if len(lemmas) >= 2:
                    suggestions.append({
                        "original": w,
                        "upgrades": list(lemmas)[:4],
                        "category": "Synonyms"
                    })

        return suggestions[:6]

    def rewrite_sentence(self, text, tone="Formal"):
        """
        Rewrite text into different stylistic tones.
        """
        if not text or not text.strip():
            return ""

        tone = tone.lower()
        rewritten = text.strip()

        # Formal replacements
        formal_rules = [
            (r"\bgonna\b", "going to"),
            (r"\bwanna\b", "wish to"),
            (r"\bgotta\b", "must"),
            (r"\bkinda\b", "somewhat"),
            (r"\bkids\b", "children"),
            (r"\bbad\b", "substandard"),
            (r"\bbig\b", "substantial"),
            (r"\bgood\b", "beneficial"),
            (r"\ba lot of\b", "a significant amount of"),
            (r"\blots of\b", "numerous"),
            (r"\bcan't\b", "cannot"),
            (r"\bdon't\b", "do not"),
            (r"\bwon't\b", "will not"),
            (r"\bI think\b", "It is evident that"),
            (r"\bso\b", "consequently"),
            (r"\bhelp\b", "assist")
        ]

        # Casual replacements
        casual_rules = [
            (r"\bcannot\b", "can't"),
            (r"\bdo not\b", "don't"),
            (r"\bwill not\b", "won't"),
            (r"\bgoing to\b", "gonna"),
            (r"\bwant to\b", "wanna"),
            (r"\bconsequently\b", "so"),
            (r"\bfurthermore\b", "also"),
            (r"\bassist\b", "help"),
            (r"\bsubstantial\b", "huge"),
            (r"\bbeneficial\b", "great")
        ]

        # Concise replacements
        concise_rules = [
            (r"\bdue to the fact that\b", "because"),
            (r"\bat the present time\b", "now"),
            (r"\bin order to\b", "to"),
            (r"\bhas the ability to\b", "can"),
            (r"\bmake a decision\b", "decide"),
            (r"\bconduct an investigation of\b", "investigate"),
            (r"\ba large number of\b", "many"),
            (r"\bfor the purpose of\b", "for")
        ]

        # Academic replacements
        academic_rules = [
            (r"\bshow\b", "demonstrate"),
            (r"\bfind out\b", "ascertain"),
            (r"\blook into\b", "investigate"),
            (r"\bget\b", "acquire"),
            (r"\buse\b", "utilize"),
            (r"\bbig problem\b", "critical impediment"),
            (r"\bgood results\b", "empirical outcomes"),
            (r"\bmake sure\b", "ensure")
        ]

        rules = []
        if "formal" in tone:
            rules = formal_rules
        elif "casual" in tone:
            rules = casual_rules
        elif "concise" in tone:
            rules = concise_rules
        elif "academic" in tone:
            rules = academic_rules

        for pattern, replacement in rules:
            rewritten = re.sub(pattern, replacement, rewritten, flags=re.IGNORECASE)

        # Capitalize first letter of rewritten text
        if rewritten and len(rewritten) > 1:
            rewritten = rewritten[0].upper() + rewritten[1:]

        return rewritten

    def translate_urdu(self, text):
        """
        Translates common Urdu / Roman Urdu phrases to standard English.
        """
        text_lower = text.strip().lower()
        # Direct phrase lookup
        for urdu, eng in self.urdu_dictionary.items():
            if urdu in text_lower:
                text_lower = text_lower.replace(urdu, eng)
                
        # Word-by-word fallback
        word_map = {
            "mein": "I", "main": "I", "hum": "we", "aap": "you", "tum": "you",
            "woh": "he/she/they", "hai": "is", "hain": "are", "tha": "was",
            "the": "were", "gaari": "car", "school": "school", "kitab": "book",
            "pani": "water", "khana": "food", "acha": "good", "bura": "bad",
            "bohot": "very", "bohat": "very", "shukriya": "thank you"
        }
        words = text_lower.split()
        translated_words = [word_map.get(w, w) for w in words]
        result = " ".join(translated_words)
        return result.capitalize() if result else ""
