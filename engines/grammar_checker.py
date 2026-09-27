import re
import nltk

class GrammarCheckerEngine:
    def __init__(self):
        # Common phonetic vowel exceptions for articles
        self.vowel_exceptions_a = {"hour", "honest", "heir", "honor", "honorable"}
        self.vowel_exceptions_an = {"university", "uniform", "unicorn", "unique", "unit", "european", "one"}
        
        # Subject-Verb Agreement rules
        self.sva_patterns = [
            # I / You / We / They + has -> have
            (r"\b(I|you|we|they|You|We|They)\s+(has)\b", "have", "Subject-Verb Agreement", 
             "Plural subjects and 'I'/'you' take 'have', not 'has'."),
            # He / She / It + have -> has
            (r"\b(he|she|it|He|She|It)\s+(have)\b", "has", "Subject-Verb Agreement",
             "Third-person singular subjects (he/she/it) take 'has', not 'have'."),
            # I + is / are -> am
            (r"\b(I)\s+(is|are)\b", "am", "Subject-Verb Agreement",
             "The first-person pronoun 'I' pairs with 'am'."),
            # He / She / It + are -> is
            (r"\b(he|she|it|He|She|It)\s+(are)\b", "is", "Subject-Verb Agreement",
             "Singular third-person subjects take 'is', not 'are'."),
            # We / They / You + is -> are
            (r"\b(we|they|you|We|They|You)\s+(is)\b", "are", "Subject-Verb Agreement",
             "Plural subjects and 'you' take 'are', not 'is'."),
            # He / She + base verb (e.g. He go -> He goes, She eat -> She eats)
            (r"\b(he|she|He|She)\s+(go|walk|read|write|play|run|want|like|need)\b", 
             lambda m: m.group(2) + ("es" if m.group(2).endswith(('o', 's', 'sh', 'ch', 'x')) else "s"),
             "Subject-Verb Agreement", "Singular third-person subject requires third-person singular verb (-s/-es).")
        ]

        # Auxiliary + Verb Form patterns
        self.aux_verb_patterns = [
            # am / is / are + base verb -> present participle (-ing)
            (r"\b(am|is|are|was|were)\s+(go|walk|run|study|work|write|read|eat|play|sleep)\b",
             lambda m: m.group(2) + "ing" if not m.group(2).endswith("e") else m.group(2)[:-1] + "ing",
             "Verb Form", "Continuous tenses require present participle (-ing) form of the main verb."),
            # did + past tense -> did + base form (e.g., did went -> did go)
            (r"\b(did|didn't|did not)\s+(went)\b", "go", "Verb Tense", "Use the base form of the verb after 'did'."),
            (r"\b(did|didn't|did not)\s+(saw)\b", "see", "Verb Tense", "Use the base form of the verb after 'did'."),
            (r"\b(did|didn't|did not)\s+(ate)\b", "eat", "Verb Tense", "Use the base form of the verb after 'did'."),
            (r"\b(did|didn't|did not)\s+(wrote)\b", "write", "Verb Tense", "Use the base form of the verb after 'did'."),
            (r"\b(did|didn't|did not)\s+(came)\b", "come", "Verb Tense", "Use the base form of the verb after 'did'.")
        ]

        # Preposition misuse patterns
        self.preposition_patterns = [
            (r"\b(interested)\s+(on|at|for)\b", "interested in", "Preposition", "The correct preposition for 'interested' is 'in'."),
            (r"\b(depend|depends|depending)\s+(of|in|to)\b", r"\1 on", "Preposition", "'Depend' takes the preposition 'on'."),
            (r"\b(good|bad|skilled)\s+(in)\s+(math|english|coding|sports|writing|languages)\b", r"\1 at \3", "Preposition", "Use 'good at' when expressing proficiency in a subject or skill."),
            (r"\b(congratulate|congratulations)\s+(for)\b", r"\1 on", "Preposition", "Use 'congratulate on', not 'congratulate for'."),
            (r"\b(listen)\s+(music|radio|him|her|them)\b", r"listen to \2", "Preposition", "'Listen' requires the preposition 'to' before an object."),
            (r"\b(married)\s+(with)\b", "married to", "Preposition", "The standard preposition with 'married' is 'to'.")
        ]

        # Redundancy and Double Negatives
        self.idiom_patterns = [
            (r"\b(don't|doesn't|didn't|cannot|can't)\s+(know|have|see|want)\s+(nothing|nobody|nowhere)\b",
             lambda m: f"{m.group(1)} {m.group(2)} " + ("anything" if m.group(3) == "nothing" else "anybody" if m.group(3) == "nobody" else "anywhere"),
             "Double Negative", "Avoid double negatives in English; use 'any-' words with negative auxiliaries."),
            (r"\b(can|could)\s+able to\b", "can", "Redundancy", "'Can' and 'able to' express the same meaning; avoid using both together."),
            (r"\b(more better)\b", "better", "Comparative Degree", "'Better' is already comparative. Do not use 'more better'."),
            (r"\b(more faster)\b", "faster", "Comparative Degree", "'Faster' is already comparative. Do not use 'more faster'.")
        ]

    def check(self, text):
        """
        Scan text for grammatical errors and return list of suggestions.
        """
        if not text or not text.strip():
            return []

        errors = []

        # 1. Subject-Verb Agreement checks
        for pattern, replacement, category, explanation in self.sva_patterns:
            for match in re.finditer(pattern, text):
                start, end = match.span()
                matched_text = match.group(0)
                
                if callable(replacement):
                    rep_word = replacement(match)
                    full_rep = re.sub(r"\b" + match.group(2) + r"\b", rep_word, matched_text)
                else:
                    # Replace only the verb portion
                    if len(match.groups()) >= 2:
                        full_rep = f"{match.group(1)} {replacement}"
                    else:
                        full_rep = replacement

                errors.append({
                    "id": f"grammar_sva_{start}_{end}",
                    "type": "Grammar",
                    "category": category,
                    "severity": "critical",
                    "original": matched_text,
                    "replacement": full_rep,
                    "suggestions": [full_rep],
                    "start": start,
                    "end": end,
                    "message": f"Grammar error: '{matched_text}' -> '{full_rep}'",
                    "explanation": explanation
                })

        # 2. Auxiliary + Verb Form checks
        for pattern, replacement, category, explanation in self.aux_verb_patterns:
            for match in re.finditer(pattern, text):
                start, end = match.span()
                matched_text = match.group(0)
                
                if callable(replacement):
                    verb_rep = replacement(match)
                    full_rep = f"{match.group(1)} {verb_rep}"
                else:
                    full_rep = f"{match.group(1)} {replacement}"

                errors.append({
                    "id": f"grammar_aux_{start}_{end}",
                    "type": "Grammar",
                    "category": category,
                    "severity": "critical",
                    "original": matched_text,
                    "replacement": full_rep,
                    "suggestions": [full_rep],
                    "start": start,
                    "end": end,
                    "message": f"Incorrect verb form: '{matched_text}' -> '{full_rep}'",
                    "explanation": explanation
                })

        # 3. Article checks (a vs an)
        article_matches = list(re.finditer(r"\b(a|an|A|An)\s+([A-Za-z]+)\b", text))
        for match in article_matches:
            art = match.group(1)
            next_word = match.group(2).lower()
            start, end = match.span()
            orig_match = match.group(0)

            # Rules for 'a' vs 'an'
            starts_with_vowel = next_word[0] in "aeiou"
            
            # Special exceptions
            is_an_exception_a = next_word in self.vowel_exceptions_a
            is_an_exception_an = next_word in self.vowel_exceptions_an

            should_be_an = (starts_with_vowel and not is_an_exception_an) or is_an_exception_a

            if art.lower() == "a" and should_be_an:
                correct_art = "An" if art[0].isupper() else "an"
                rep = f"{correct_art} {match.group(2)}"
                errors.append({
                    "id": f"grammar_art_{start}_{end}",
                    "type": "Grammar",
                    "category": "Article Usage",
                    "severity": "critical",
                    "original": orig_match,
                    "replacement": rep,
                    "suggestions": [rep],
                    "start": start,
                    "end": end,
                    "message": f"Article error: Use '{correct_art}' before vowel sounds ('{next_word}').",
                    "explanation": f"Use '{correct_art}' instead of '{art}' because '{next_word}' begins with a vowel sound."
                })
            elif art.lower() == "an" and not should_be_an:
                correct_art = "A" if art[0].isupper() else "a"
                rep = f"{correct_art} {match.group(2)}"
                errors.append({
                    "id": f"grammar_art_{start}_{end}",
                    "type": "Grammar",
                    "category": "Article Usage",
                    "severity": "critical",
                    "original": orig_match,
                    "replacement": rep,
                    "suggestions": [rep],
                    "start": start,
                    "end": end,
                    "message": f"Article error: Use '{correct_art}' before consonant sounds ('{next_word}').",
                    "explanation": f"Use '{correct_art}' instead of '{art}' because '{next_word}' begins with a consonant sound."
                })

        # 4. Preposition checks
        for pattern, replacement, category, explanation in self.preposition_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start, end = match.span()
                orig_match = match.group(0)
                rep = re.sub(pattern, replacement, orig_match, flags=re.IGNORECASE)
                if rep.lower() != orig_match.lower():
                    errors.append({
                        "id": f"grammar_prep_{start}_{end}",
                        "type": "Grammar",
                        "category": category,
                        "severity": "critical",
                        "original": orig_match,
                        "replacement": rep,
                        "suggestions": [rep],
                        "start": start,
                        "end": end,
                        "message": f"Preposition suggestion: '{orig_match}' -> '{rep}'",
                        "explanation": explanation
                    })

        # 5. Idioms, Double Negatives, and Redundancies
        for pattern, replacement, category, explanation in self.idiom_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start, end = match.span()
                orig_match = match.group(0)
                if callable(replacement):
                    rep = replacement(match)
                else:
                    rep = replacement

                errors.append({
                    "id": f"grammar_idiom_{start}_{end}",
                    "type": "Grammar",
                    "category": category,
                    "severity": "critical",
                    "original": orig_match,
                    "replacement": rep,
                    "suggestions": [rep],
                    "start": start,
                    "end": end,
                    "message": f"Stylistic / Grammatical improvement: '{orig_match}' -> '{rep}'",
                    "explanation": explanation
                })

        # 6. Repeated consecutive words (e.g., "in in", "the the")
        for match in re.finditer(r"\b([A-Za-z]+)\s+\1\b", text, re.IGNORECASE):
            start, end = match.span()
            repeated_word = match.group(1)
            errors.append({
                "id": f"grammar_repeat_{start}_{end}",
                "type": "Grammar",
                "category": "Repeated Word",
                "severity": "warning",
                "original": match.group(0),
                "replacement": repeated_word,
                "suggestions": [repeated_word],
                "start": start,
                "end": end,
                "message": f"Repeated word detected: '{repeated_word}'.",
                "explanation": f"The word '{repeated_word}' is repeated consecutively. Consider removing the duplicate."
            })

        # 7. Standalone lowercase 'i' -> 'I'
        for match in re.finditer(r"(?<![A-Za-z])i(?![A-Za-z])", text):
            start, end = match.span()
            errors.append({
                "id": f"grammar_cap_i_{start}_{end}",
                "type": "Grammar",
                "category": "Capitalization",
                "severity": "warning",
                "original": "i",
                "replacement": "I",
                "suggestions": ["I"],
                "start": start,
                "end": end,
                "message": "Capitalize personal pronoun 'I'.",
                "explanation": "In English, the personal pronoun 'I' must always be capitalized."
            })

        # Deduplicate overlapping errors and sort by starting offset
        sorted_errors = sorted(errors, key=lambda x: (x["start"], -(x["end"] - x["start"])))
        deduped = []
        last_end = -1
        for err in sorted_errors:
            if err["start"] >= last_end:
                deduped.append(err)
                last_end = err["end"]

        return deduped

    def auto_correct(self, text):
        """
        Auto-corrects all detected grammatical errors.
        """
        errors = self.check(text)
        if not errors:
            return text

        chars = list(text)
        for err in sorted(errors, key=lambda x: x["start"], reverse=True):
            chars[err["start"]:err["end"]] = list(err["replacement"])

        return "".join(chars)
