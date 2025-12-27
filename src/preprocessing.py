"""
Text Preprocessing Module

Handles all text normalization, tokenization, and cleaning operations
using NLTK and regex for robust text preprocessing.

Author: Tharun Ponnam
"""

import re
import string
from typing import List, Optional, Union

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize


class TextPreprocessor:
    """
    Comprehensive text preprocessing pipeline for NLP tasks.
    
    This class provides methods for cleaning, normalizing, and transforming
    raw text data into a format suitable for machine learning models.
    
    Attributes:
        remove_stopwords (bool): Whether to remove stopwords
        use_lemmatization (bool): Whether to apply lemmatization
        use_stemming (bool): Whether to apply stemming
        lowercase (bool): Whether to convert text to lowercase
    """
    
    def __init__(
        self,
        remove_stopwords: bool = True,
        use_lemmatization: bool = True,
        use_stemming: bool = False,
        lowercase: bool = True,
        custom_stopwords: Optional[List[str]] = None
    ):
        """
        Initialize the TextPreprocessor with specified options.
        
        Args:
            remove_stopwords: Flag to remove English stopwords
            use_lemmatization: Flag to apply WordNet lemmatization
            use_stemming: Flag to apply Porter stemming
            lowercase: Flag to convert all text to lowercase
            custom_stopwords: Additional stopwords to remove
        """
        self._download_nltk_resources()
        
        self.remove_stopwords = remove_stopwords
        self.use_lemmatization = use_lemmatization
        self.use_stemming = use_stemming
        self.lowercase = lowercase
        
        self.stop_words = set(stopwords.words('english'))
        if custom_stopwords:
            self.stop_words.update(custom_stopwords)
        
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        
        self._url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self._email_pattern = re.compile(r'\b[\w.-]+@[\w.-]+\.\w+\b')
        self._phone_pattern = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
        self._number_pattern = re.compile(r'\b\d+\b')
        self._whitespace_pattern = re.compile(r'\s+')
        self._special_chars = re.compile(r'[^\w\s]')
    
    @staticmethod
    def _download_nltk_resources():
        """Download required NLTK resources if not present."""
        resources = [
            'punkt',
            'punkt_tab',
            'stopwords',
            'wordnet',
            'averaged_perceptron_tagger',
            'averaged_perceptron_tagger_eng'
        ]
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                try:
                    nltk.download(resource, quiet=True)
                except Exception:
                    pass
    
    def clean_text(self, text: str) -> str:
        """
        Apply all cleaning operations to raw text.
        
        Args:
            text: Raw input text string
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        text = self._url_pattern.sub(' ', text)
        text = self._email_pattern.sub(' ', text)
        text = self._phone_pattern.sub(' ', text)
        
        if self.lowercase:
            text = text.lower()
        
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = self._number_pattern.sub(' ', text)
        text = self._whitespace_pattern.sub(' ', text)
        
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into individual words.
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens
        """
        cleaned = self.clean_text(text)
        tokens = word_tokenize(cleaned)
        
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]
        
        if self.use_lemmatization:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        elif self.use_stemming:
            tokens = [self.stemmer.stem(t) for t in tokens]
        
        tokens = [t for t in tokens if len(t) > 1]
        
        return tokens
    
    def preprocess(self, text: str) -> str:
        """
        Full preprocessing pipeline returning joined tokens.
        
        Args:
            text: Raw input text
            
        Returns:
            Preprocessed text as single string
        """
        tokens = self.tokenize(text)
        return ' '.join(tokens)
    
    def batch_preprocess(
        self, 
        texts: List[str], 
        show_progress: bool = False
    ) -> List[str]:
        """
        Process multiple texts in batch.
        
        Args:
            texts: List of raw text strings
            show_progress: Whether to display progress bar
            
        Returns:
            List of preprocessed text strings
        """
        if show_progress:
            try:
                from tqdm import tqdm
                return [self.preprocess(t) for t in tqdm(texts, desc="Preprocessing")]
            except ImportError:
                pass
        
        return [self.preprocess(t) for t in texts]
    
    def extract_sentences(self, text: str) -> List[str]:
        """
        Split text into individual sentences.
        
        Args:
            text: Input text string
            
        Returns:
            List of sentence strings
        """
        return sent_tokenize(text)
    
    def get_word_frequency(self, text: str) -> dict:
        """
        Calculate word frequencies in text.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary mapping words to their frequencies
        """
        tokens = self.tokenize(text)
        freq = {}
        for token in tokens:
            freq[token] = freq.get(token, 0) + 1
        return dict(sorted(freq.items(), key=lambda x: x[1], reverse=True))
    
    def get_pos_tags(self, text: str) -> List[tuple]:
        """
        Get part-of-speech tags for tokens.
        
        Args:
            text: Input text string
            
        Returns:
            List of (token, POS_tag) tuples
        """
        cleaned = self.clean_text(text)
        tokens = word_tokenize(cleaned)
        return nltk.pos_tag(tokens)
    
    def __repr__(self) -> str:
        return (
            f"TextPreprocessor(remove_stopwords={self.remove_stopwords}, "
            f"use_lemmatization={self.use_lemmatization}, "
            f"use_stemming={self.use_stemming})"
        )
