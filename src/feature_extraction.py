"""
Feature Extraction Module

Provides various text vectorization and feature extraction methods
including TF-IDF, Count Vectorization, and N-gram features.

Author: Tharun Ponnam
"""

from typing import Optional, Tuple, Union, List
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    CountVectorizer,
    HashingVectorizer
)
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.preprocessing import StandardScaler


class FeatureExtractor:
    """
    Text feature extraction using multiple vectorization strategies.
    
    Supports TF-IDF, Bag-of-Words, and Hashing vectorization with
    optional dimensionality reduction.
    
    Attributes:
        vectorizer_type (str): Type of vectorizer to use
        max_features (int): Maximum number of features
        ngram_range (tuple): Range of n-grams to extract
    """
    
    VECTORIZER_TYPES = ['tfidf', 'count', 'hash']
    
    def __init__(
        self,
        vectorizer_type: str = 'tfidf',
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        use_idf: bool = True,
        sublinear_tf: bool = True,
        norm: str = 'l2'
    ):
        """
        Initialize feature extractor with specified parameters.
        
        Args:
            vectorizer_type: One of 'tfidf', 'count', or 'hash'
            max_features: Maximum vocabulary size
            ngram_range: Tuple of (min_n, max_n) for n-gram extraction
            min_df: Minimum document frequency for terms
            max_df: Maximum document frequency ratio
            use_idf: Whether to use inverse document frequency
            sublinear_tf: Apply sublinear TF scaling
            norm: Normalization method ('l1', 'l2', or None)
        """
        if vectorizer_type not in self.VECTORIZER_TYPES:
            raise ValueError(
                f"vectorizer_type must be one of {self.VECTORIZER_TYPES}"
            )
        
        self.vectorizer_type = vectorizer_type
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        
        self._vectorizer = self._create_vectorizer(
            vectorizer_type, max_features, ngram_range,
            min_df, max_df, use_idf, sublinear_tf, norm
        )
        
        self._dim_reducer = None
        self._scaler = None
        self._is_fitted = False
    
    def _create_vectorizer(
        self,
        vectorizer_type: str,
        max_features: int,
        ngram_range: Tuple[int, int],
        min_df: int,
        max_df: float,
        use_idf: bool,
        sublinear_tf: bool,
        norm: str
    ):
        """Create the appropriate vectorizer instance."""
        if vectorizer_type == 'tfidf':
            return TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=min_df,
                max_df=max_df,
                use_idf=use_idf,
                sublinear_tf=sublinear_tf,
                norm=norm,
                lowercase=False,
                token_pattern=r'\b\w+\b'
            )
        elif vectorizer_type == 'count':
            return CountVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=min_df,
                max_df=max_df,
                lowercase=False,
                token_pattern=r'\b\w+\b'
            )
        else:
            return HashingVectorizer(
                n_features=max_features,
                ngram_range=ngram_range,
                norm=norm,
                lowercase=False,
                token_pattern=r'\b\w+\b'
            )
    
    def fit(self, texts: List[str]) -> 'FeatureExtractor':
        """
        Fit the vectorizer on training texts.
        
        Args:
            texts: List of preprocessed text strings
            
        Returns:
            Self for method chaining
        """
        if self.vectorizer_type == 'hash':
            self._is_fitted = True
        else:
            self._vectorizer.fit(texts)
            self._is_fitted = True
        return self
    
    def transform(self, texts: List[str]) -> csr_matrix:
        """
        Transform texts to feature matrix.
        
        Args:
            texts: List of preprocessed text strings
            
        Returns:
            Sparse feature matrix
        """
        if not self._is_fitted and self.vectorizer_type != 'hash':
            raise RuntimeError("FeatureExtractor must be fitted before transform")
        
        features = self._vectorizer.transform(texts)
        
        if self._dim_reducer is not None:
            features = self._dim_reducer.transform(features)
        
        return features
    
    def fit_transform(self, texts: List[str]) -> csr_matrix:
        """
        Fit vectorizer and transform texts in one step.
        
        Args:
            texts: List of preprocessed text strings
            
        Returns:
            Sparse feature matrix
        """
        if self.vectorizer_type == 'hash':
            self._is_fitted = True
            return self._vectorizer.transform(texts)
        
        features = self._vectorizer.fit_transform(texts)
        self._is_fitted = True
        return features
    
    def reduce_dimensions(
        self,
        features: csr_matrix,
        n_components: int = 100,
        method: str = 'svd'
    ) -> np.ndarray:
        """
        Apply dimensionality reduction to features.
        
        Args:
            features: Sparse feature matrix
            n_components: Number of dimensions to reduce to
            method: Reduction method ('svd' or 'nmf')
            
        Returns:
            Dense reduced feature matrix
        """
        if method == 'svd':
            self._dim_reducer = TruncatedSVD(
                n_components=n_components,
                random_state=42
            )
        elif method == 'nmf':
            self._dim_reducer = NMF(
                n_components=n_components,
                random_state=42,
                max_iter=500
            )
        else:
            raise ValueError("method must be 'svd' or 'nmf'")
        
        reduced = self._dim_reducer.fit_transform(features)
        return reduced
    
    def get_feature_names(self) -> List[str]:
        """
        Get the vocabulary of the fitted vectorizer.
        
        Returns:
            List of feature names
        """
        if not self._is_fitted:
            raise RuntimeError("FeatureExtractor must be fitted first")
        
        if self.vectorizer_type == 'hash':
            raise ValueError("HashingVectorizer does not store feature names")
        
        return self._vectorizer.get_feature_names_out().tolist()
    
    def get_top_features(
        self,
        features: csr_matrix,
        labels: np.ndarray,
        n_top: int = 20
    ) -> dict:
        """
        Get top features for each class based on average TF-IDF.
        
        Args:
            features: Feature matrix
            labels: Class labels
            n_top: Number of top features to return
            
        Returns:
            Dictionary mapping class labels to top features
        """
        feature_names = self.get_feature_names()
        unique_labels = np.unique(labels)
        
        top_features = {}
        for label in unique_labels:
            mask = labels == label
            class_features = features[mask].mean(axis=0).A1
            top_indices = class_features.argsort()[-n_top:][::-1]
            top_features[label] = [
                (feature_names[i], class_features[i])
                for i in top_indices
            ]
        
        return top_features
    
    @property
    def vocabulary_size(self) -> int:
        """Get the size of the learned vocabulary."""
        if not self._is_fitted:
            return 0
        if self.vectorizer_type == 'hash':
            return self.max_features
        return len(self._vectorizer.vocabulary_)
    
    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"FeatureExtractor(type='{self.vectorizer_type}', "
            f"max_features={self.max_features}, "
            f"ngram_range={self.ngram_range}, {status})"
        )
