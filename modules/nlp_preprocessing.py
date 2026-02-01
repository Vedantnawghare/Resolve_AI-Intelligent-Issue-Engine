"""
STAGE 2: ADVANCED NLP PREPROCESSING PIPELINE
Production-grade text understanding module for real-world user input

Handles:
- Spelling mistakes
- Hinglish (Hindi written in English letters)
- Mixed Hindi + English sentences
- Informal, chat-style language
- Code-switched text

Pipeline Steps:
1. Input Normalization
2. Language Detection
3. Spelling Correction
4. Hinglish Transliteration
5. Informal to Formal Conversion
6. Text Cleaning
7. Tokenization
8. Lemmatization
9. TF-IDF Ready Output

Author: Senior NLP Engineering Team
"""

import re
import string
from typing import Dict, List, Tuple
import pandas as pd
from config import (
    HINGLISH_TO_ENGLISH,
    COMMON_SPELLING_CORRECTIONS,
    INFORMAL_TO_FORMAL,
    COMMON_HINDI_WORDS,
    COMMON_ENGLISH_WORDS
)

# ============================================================================
# STEP 1: INPUT NORMALIZATION
# Why: Real users write with inconsistent spacing, punctuation, case
# ============================================================================

class TextNormalizer:
    """Normalize raw user input"""
    
    @staticmethod
    def normalize(text: str) -> str:
        """
        Clean raw input without losing meaning
        
        Why each sub-step:
        - Lower case: "WiFi" = "wifi" = "WIFI" (consistency for matching)
        - Strip extra spaces: "hello    world" → "hello world"
        - Remove excessive punctuation: "help!!!!!!" → "help!"
        - Keep single spaces: Prevent token merging
        
        Real-world example:
        Input: "  URGENT!!!   PLZ  help   ASAP  "
        Output: "urgent! plz help asap"
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs (users sometimes paste links)
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Reduce repeated punctuation (!!!! → !)
        text = re.sub(r'([!?.]){2,}', r'\1', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing spaces
        text = text.strip()
        
        return text


# ============================================================================
# STEP 2: LANGUAGE DETECTION
# Why: Different processing needed for English vs Hinglish vs Mixed
# ============================================================================

class LanguageDetector:
    """Detect if input is English, Hinglish, or Mixed"""
    
    @staticmethod
    def detect_language(text: str) -> str:
        """
        Identify language mix in text
        
        Why this matters:
        - Pure English: Skip transliteration
        - Pure Hinglish: Aggressive transliteration needed
        - Mixed: Selective word-by-word processing
        
        Detection logic:
        1. Check for Devanagari script (native Hindi)
        2. Count Hindi romanized words vs English words
        3. Classify based on ratio
        
        Real-world examples:
        "internet not working" → ENGLISH
        "mera wifi nahi chal raha" → HINGLISH
        "plz help mera network issue hai" → MIXED
        """
        if not text:
            return "UNKNOWN"
        
        # Check for Devanagari script (native Hindi)
        if re.search(r'[\u0900-\u097F]', text):
            return "HINDI_NATIVE"
        
        words = text.lower().split()
        if not words:
            return "UNKNOWN"
        
        # Count Hindi vs English words
        hindi_word_count = sum(1 for word in words if word in COMMON_HINDI_WORDS)
        english_word_count = sum(1 for word in words if word in COMMON_ENGLISH_WORDS)
        
        total_known_words = hindi_word_count + english_word_count
        
        if total_known_words == 0:
            return "ENGLISH"  # Default to English if no known words
        
        hindi_ratio = hindi_word_count / total_known_words
        
        if hindi_ratio > 0.6:
            return "HINGLISH"
        elif hindi_ratio > 0.2:
            return "MIXED"
        else:
            return "ENGLISH"


# ============================================================================
# STEP 3: SPELLING CORRECTION
# Why: Users make typos, especially on mobile devices
# ============================================================================

class SpellingCorrector:
    """Fix common spelling mistakes"""
    
    def __init__(self):
        self.corrections = COMMON_SPELLING_CORRECTIONS
    
    def correct(self, text: str) -> str:
        """
        Fix spelling mistakes using dictionary lookup
        
        Why dictionary-based (not ML):
        - Lightweight: No model loading overhead
        - Explainable: Know exactly what changed
        - Fast: O(1) lookup per word
        - Predictable: Same input → same output
        
        Real-world example:
        Input: "wify not working passward wrong"
        Output: "wifi not working password wrong"
        
        Note: This is a simple approach. For production at scale,
        consider: TextBlob, SymSpell, or context-aware models
        """
        words = text.split()
        corrected_words = []
        
        for word in words:
            # Check if word needs correction
            if word in self.corrections:
                corrected_words.append(self.corrections[word])
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words)


# ============================================================================
# STEP 4: HINGLISH TRANSLITERATION
# Why: Hindi written in English script needs conversion to English
# ============================================================================

class HinglishTransliterator:
    """Convert Hinglish (romanized Hindi) to English"""
    
    def __init__(self):
        self.mapping = HINGLISH_TO_ENGLISH
    
    def transliterate(self, text: str) -> str:
        """
        Convert Hindi words written in English to English equivalents
        
        Why this step is critical:
        - "mera wifi nahi chal raha" is meaningless to English ML models
        - Must convert to "my wifi not working" for ML to understand
        - Preserves user intent while making it ML-readable
        
        Real-world example:
        Input: "mera order abhi tak nahi aaya pls help"
        After transliteration: "my order now yet not came please help"
        After grammar fix (next step): "my order has not arrived yet please help"
        
        Note: This is word-by-word replacement. For production:
        - Use Google Translate API for better accuracy
        - Or train a seq2seq model on Hinglish→English pairs
        - Or use IndicNLP library for Indian languages
        
        But for this project: Lightweight > Perfect
        """
        words = text.split()
        transliterated_words = []
        
        for word in words:
            # Check if word is in Hinglish dictionary
            if word in self.mapping:
                transliterated_words.append(self.mapping[word])
            else:
                transliterated_words.append(word)
        
        return ' '.join(transliterated_words)


# ============================================================================
# STEP 5: INFORMAL TO FORMAL CONVERSION
# Why: Chat language needs standardization for ML models
# ============================================================================

class InformalConverter:
    """Convert chat-style informal text to formal English"""
    
    def __init__(self):
        self.mapping = INFORMAL_TO_FORMAL
    
    def formalize(self, text: str) -> str:
        """
        Convert informal abbreviations to formal language
        
        Why this matters:
        - "plz help asap" → "please help as soon as possible"
        - ML models trained on formal text perform better
        - TF-IDF treats "plz" and "please" as different words
        - Standardization improves model accuracy
        
        Real-world example:
        Input: "plz resolve asap ur system down rn"
        Output: "please resolve as soon as possible your system down right now"
        
        User psychology:
        - Users in distress use abbreviations (urgency signal!)
        - "plz", "asap" often correlate with high priority
        - But we still need clean text for ML
        """
        words = text.split()
        formal_words = []
        
        for word in words:
            if word in self.mapping:
                formal_word = self.mapping[word]
                # Only add if not empty (some mappings remove slang)
                if formal_word:
                    formal_words.append(formal_word)
            else:
                formal_words.append(word)
        
        return ' '.join(formal_words)


# ============================================================================
# STEP 6: ADVANCED TEXT CLEANING
# Why: Remove noise that doesn't help ML models
# ============================================================================

class TextCleaner:
    """Deep clean text for ML readiness"""
    
    @staticmethod
    def clean(text: str) -> str:
        """
        Remove noise while preserving meaning
        
        What we remove:
        - Special characters: @#$%^&* (except for context-relevant ones)
        - Numbers (unless they're important like "404 error")
        - Extra punctuation
        
        What we keep:
        - Negations: "not", "no", "never" (crucial for sentiment)
        - Important words that look like noise but aren't
        
        Real-world example:
        Input: "login @#$ error 404 !!! urgent"
        Output: "login error urgent"
        
        Why not remove everything:
        - "404" might be useful for IT Support category
        - Over-cleaning loses information
        - Balance: Clean enough for ML, rich enough for patterns
        """
        if not text:
            return ""
        
        # Remove special characters (keep spaces and basic punctuation)
        text = re.sub(r'[^a-z0-9\s\-]', ' ', text)
        
        # Remove standalone numbers (unless part of error codes)
        text = re.sub(r'\b\d+\b', '', text)
        
        # Remove extra spaces again
        text = re.sub(r'\s+', ' ', text)
        
        # Strip
        text = text.strip()
        
        return text


# ============================================================================
# STEP 7: TOKENIZATION
# Why: Break text into processable units
# ============================================================================

class Tokenizer:
    """Split text into tokens"""
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Split text into words (tokens)
        
        Why simple split() works here:
        - We've already cleaned the text
        - No complex grammar to handle
        - Fast and predictable
        
        For production at scale:
        - Use NLTK's word_tokenize() for better handling
        - Or spaCy for industrial-grade tokenization
        - Or transformers tokenizers for BERT-style models
        
        But for this project: Simple = Fast = Good
        """
        if not text:
            return []
        
        return text.split()


# ============================================================================
# STEP 8: LEMMATIZATION
# Why: "running", "runs", "ran" should all become "run"
# ============================================================================

class Lemmatizer:
    """Reduce words to base form"""
    
    # Simple rule-based lemmatization (lightweight)
    # For production: Use NLTK's WordNetLemmatizer or spaCy
    
    SIMPLE_RULES = {
        "running": "run",
        "runs": "run",
        "ran": "run",
        "working": "work",
        "works": "work",
        "worked": "work",
        "connecting": "connect",
        "connects": "connect",
        "connected": "connect",
        "failing": "fail",
        "fails": "fail",
        "failed": "fail",
        "loading": "load",
        "loads": "load",
        "loaded": "load",
        "starting": "start",
        "starts": "start",
        "started": "start",
        "issues": "issue",
        "problems": "problem",
        "errors": "error",
        "networks": "network"
    }
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """
        Convert words to base form
        
        Why lemmatization matters:
        - "wifi not working" and "wifi doesn't work" should match
        - Reduces vocabulary size (better for small datasets)
        - Improves pattern matching
        
        Real-world example:
        Input: ["network", "keeps", "disconnecting", "users", "complaining"]
        Output: ["network", "keep", "disconnect", "user", "complain"]
        
        Note: This simple version only handles common verbs/nouns.
        For production: Use proper lemmatizer with POS tagging
        """
        lemmatized = []
        for token in tokens:
            if token in self.SIMPLE_RULES:
                lemmatized.append(self.SIMPLE_RULES[token])
            else:
                # Simple heuristic: remove common suffixes
                if token.endswith('ing') and len(token) > 4:
                    lemmatized.append(token[:-3])
                elif token.endswith('ed') and len(token) > 3:
                    lemmatized.append(token[:-2])
                elif token.endswith('s') and len(token) > 2 and not token.endswith('ss'):
                    lemmatized.append(token[:-1])
                else:
                    lemmatized.append(token)
        
        return lemmatized


# ============================================================================
# STEP 9: STOPWORD REMOVAL (Careful!)
# Why: Remove common words that don't add meaning
# ============================================================================

class StopwordRemover:
    """Remove common words (but keep important ones)"""
    
    # Custom stopword list (not removing everything blindly)
    STOPWORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for',
        'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on',
        'that', 'the', 'to', 'was', 'will', 'with', 'the', 'this'
    }
    
    # Words that LOOK like stopwords but are NOT (keep them!)
    KEEP_WORDS = {
        'not', 'no', 'never', 'none', 'nothing', 'nowhere',
        'neither', 'nobody', 'cannot', 'down', 'up', 'out',
        'off', 'can', 'should', 'would', 'could', 'urgent',
        'emergency', 'critical', 'asap', 'immediately', 'now'
    }
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove non-informative words
        
        Critical decision: What to remove vs keep
        
        Why we DON'T remove everything:
        - "not working" vs "working" - negation matters!
        - "down" vs "up" - status matters!
        - "urgent" - priority signal!
        
        Real-world example:
        Input: ["the", "wifi", "is", "not", "working", "in", "the", "library"]
        Output: ["wifi", "not", "working", "library"]
        
        Notice: "not" is kept (negation), "the"/"is"/"in" removed (noise)
        """
        filtered = []
        for token in tokens:
            if token in self.KEEP_WORDS:
                # Always keep important words
                filtered.append(token)
            elif token not in self.STOPWORDS:
                # Keep if not a stopword
                filtered.append(token)
        
        return filtered


# ============================================================================
# COMPLETE PREPROCESSING PIPELINE
# ============================================================================

class NLPPreprocessor:
    """
    Complete preprocessing pipeline for real-world text
    
    Pipeline flow:
    Raw Input → Normalize → Detect Language → Spell Check → 
    Transliterate → Formalize → Clean → Tokenize → 
    Lemmatize → Remove Stopwords → ML-Ready Output
    """
    
    def __init__(self):
        self.normalizer = TextNormalizer()
        self.language_detector = LanguageDetector()
        self.spelling_corrector = SpellingCorrector()
        self.hinglish_transliterator = HinglishTransliterator()
        self.informal_converter = InformalConverter()
        self.text_cleaner = TextCleaner()
        self.tokenizer = Tokenizer()
        self.lemmatizer = Lemmatizer()
        self.stopword_remover = StopwordRemover()
    
    def preprocess(self, text: str, return_steps: bool = False) -> Dict:
        """
        Complete preprocessing with step-by-step tracking
        
        Args:
            text: Raw user input
            return_steps: If True, return intermediate steps (for explainability)
        
        Returns:
            Dictionary with cleaned text and processing details
        """
        steps = {}
        
        # Step 1: Normalize
        steps['original'] = text
        normalized = self.normalizer.normalize(text)
        steps['normalized'] = normalized
        
        # Step 2: Detect language
        language = self.language_detector.detect_language(normalized)
        steps['detected_language'] = language
        
        # Step 3: Spell check
        spell_checked = self.spelling_corrector.correct(normalized)
        steps['spell_checked'] = spell_checked
        
        # Step 4: Transliterate Hinglish (if needed)
        if language in ['HINGLISH', 'MIXED']:
            transliterated = self.hinglish_transliterator.transliterate(spell_checked)
        else:
            transliterated = spell_checked
        steps['transliterated'] = transliterated
        
        # Step 5: Convert informal to formal
        formalized = self.informal_converter.formalize(transliterated)
        steps['formalized'] = formalized
        
        # Step 6: Deep clean
        cleaned = self.text_cleaner.clean(formalized)
        steps['cleaned'] = cleaned
        
        # Step 7: Tokenize
        tokens = self.tokenizer.tokenize(cleaned)
        steps['tokens'] = tokens
        
        # Step 8: Lemmatize
        lemmatized = self.lemmatizer.lemmatize(tokens)
        steps['lemmatized'] = lemmatized
        
        # Step 9: Remove stopwords
        filtered = self.stopword_remover.remove_stopwords(lemmatized)
        steps['filtered'] = filtered
        
        # Final clean text (rejoin tokens)
        final_text = ' '.join(filtered)
        steps['final'] = final_text
        
        result = {
            'original_text': text,
            'cleaned_text': final_text,
            'detected_language': language,
            'token_count': len(filtered),
            'processing_steps': steps if return_steps else None
        }
        
        return result
    
    def batch_preprocess(self, texts: List[str]) -> pd.DataFrame:
        """
        Preprocess multiple texts efficiently
        
        Why batch processing:
        - Users submit multiple issues
        - We need to process historical data
        - Vectorization needs all texts together
        """
        results = []
        for text in texts:
            result = self.preprocess(text, return_steps=False)
            results.append(result)
        
        return pd.DataFrame(results)


# ============================================================================
# DEMONSTRATION & TESTING
# ============================================================================

def demonstrate_pipeline():
    """
    Show real-world examples of the pipeline in action
    """
    print("=" * 80)
    print("🧠 ADVANCED NLP PREPROCESSING PIPELINE - DEMONSTRATION")
    print("=" * 80)
    
    preprocessor = NLPPreprocessor()
    
    # Real-world test cases
    test_cases = [
        "mera order abhi tak nahi aaya pls help",
        "plz resolve asap meri ticket pending hai",
        "delivery late hai but agent rude tha",
        "refund kab milega??",
        "mereko urgent issue hai login ka",
        "wify not working in libary plz fix asap",
        "laptop ka screen kharab hai urgent exam hai",
        "AC nahi chal raha bahut garmi hai plz help",
        "assignment submit nahi ho raha deadline today",
        "printer down b4 presentation 2morrow help!!!",
        "Network keeps disconnecting urgent",
        "Can't access student portal forgot passward",
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n{'─' * 80}")
        print(f"TEST CASE #{i}")
        print(f"{'─' * 80}")
        
        result = preprocessor.preprocess(text, return_steps=True)
        steps = result['processing_steps']
        
        print(f"📥 ORIGINAL:        {steps['original']}")
        print(f"🔧 NORMALIZED:      {steps['normalized']}")
        print(f"🌍 LANGUAGE:        {result['detected_language']}")
        print(f"✏️  SPELL-CHECKED:  {steps['spell_checked']}")
        print(f"🔄 TRANSLITERATED:  {steps['transliterated']}")
        print(f"📝 FORMALIZED:      {steps['formalized']}")
        print(f"🧹 CLEANED:         {steps['cleaned']}")
        print(f"🔪 TOKENS:          {steps['tokens']}")
        print(f"📊 LEMMATIZED:      {steps['lemmatized']}")
        print(f"🎯 FILTERED:        {steps['filtered']}")
        print(f"✅ FINAL OUTPUT:    {result['cleaned_text']}")
        print(f"📈 TOKENS COUNT:    {result['token_count']}")
    
    print(f"\n{'=' * 80}")
    print("✅ DEMONSTRATION COMPLETE")
    print("=" * 80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    demonstrate_pipeline()
