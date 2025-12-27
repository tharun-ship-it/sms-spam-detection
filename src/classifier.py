"""
SMS Spam Detection - Main Classifier Module

Unified interface for spam detection combining preprocessing,
feature extraction, and model training into a single pipeline.

Author: Tharun Ponnam
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .preprocessing import TextPreprocessor
from .feature_extraction import FeatureExtractor
from .model_trainer import ModelTrainer, ModelMetrics, EnsembleTrainer


class WordsClassifier:
    """
    End-to-end text classification pipeline.
    
    Provides a complete workflow from raw text to predictions,
    including preprocessing, feature extraction, and model training.
    
    Example:
        >>> classifier = WordsClassifier(model_type='naive_bayes')
        >>> classifier.fit(train_texts, train_labels)
        >>> predictions = classifier.predict(test_texts)
        >>> metrics = classifier.evaluate(test_texts, test_labels)
        >>> print(f"Accuracy: {metrics.accuracy:.4f}")
    
    Attributes:
        preprocessor: TextPreprocessor instance
        extractor: FeatureExtractor instance
        trainer: ModelTrainer instance
    """
    
    def __init__(
        self,
        model_type: str = 'naive_bayes',
        vectorizer_type: str = 'tfidf',
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        remove_stopwords: bool = True,
        use_lemmatization: bool = True,
        **model_kwargs
    ):
        """
        Initialize the classifier with specified components.
        
        Args:
            model_type: Type of classifier ('naive_bayes', 'svm', 
                       'random_forest', 'logistic', etc.)
            vectorizer_type: Type of vectorizer ('tfidf', 'count', 'hash')
            max_features: Maximum vocabulary size
            ngram_range: Range for n-gram extraction
            remove_stopwords: Whether to remove stopwords
            use_lemmatization: Whether to apply lemmatization
            **model_kwargs: Additional arguments for the model
        """
        self.model_type = model_type
        self.vectorizer_type = vectorizer_type
        
        self.preprocessor = TextPreprocessor(
            remove_stopwords=remove_stopwords,
            use_lemmatization=use_lemmatization
        )
        
        self.extractor = FeatureExtractor(
            vectorizer_type=vectorizer_type,
            max_features=max_features,
            ngram_range=ngram_range
        )
        
        self.trainer = ModelTrainer(model_type=model_type, **model_kwargs)
        
        self._classes = None
        self._is_fitted = False
        self._training_metadata = {}
    
    def fit(
        self,
        texts: Union[List[str], pd.Series],
        labels: Union[List[str], np.ndarray, pd.Series],
        validation_split: float = 0.0,
        verbose: bool = True
    ) -> 'WordsClassifier':
        """
        Train the classifier on given data.
        
        Args:
            texts: List of raw text strings
            labels: Corresponding labels for each text
            validation_split: Fraction of data for validation
            verbose: Whether to print progress information
            
        Returns:
            Self for method chaining
        """
        start_time = datetime.now()
        
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        if isinstance(labels, pd.Series):
            labels = labels.values
        
        labels = np.asarray(labels)
        self._classes = np.unique(labels)
        
        if verbose:
            print(f"Training WordsClassifier ({self.model_type})")
            print(f"  Samples: {len(texts)}")
            print(f"  Classes: {len(self._classes)}")
        
        if verbose:
            print("  Preprocessing texts...")
        processed_texts = self.preprocessor.batch_preprocess(texts)
        
        if verbose:
            print("  Extracting features...")
        features = self.extractor.fit_transform(processed_texts)
        
        if verbose:
            print(f"  Feature matrix: {features.shape}")
            print("  Training model...")
        
        self.trainer.fit(features, labels)
        
        self._is_fitted = True
        self._training_metadata = {
            'n_samples': len(texts),
            'n_classes': len(self._classes),
            'n_features': features.shape[1],
            'training_time': str(datetime.now() - start_time),
            'model_type': self.model_type,
            'vectorizer_type': self.vectorizer_type
        }
        
        if verbose:
            print(f"  Training complete in {self._training_metadata['training_time']}")
        
        return self
    
    def predict(
        self,
        texts: Union[List[str], pd.Series, str]
    ) -> np.ndarray:
        """
        Predict labels for new texts.
        
        Args:
            texts: Text(s) to classify
            
        Returns:
            Predicted labels
        """
        self._check_fitted()
        
        if isinstance(texts, str):
            texts = [texts]
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        processed = self.preprocessor.batch_preprocess(texts)
        features = self.extractor.transform(processed)
        
        return self.trainer.predict(features)
    
    def predict_proba(
        self,
        texts: Union[List[str], pd.Series, str]
    ) -> np.ndarray:
        """
        Get prediction probabilities for texts.
        
        Args:
            texts: Text(s) to classify
            
        Returns:
            Probability array
        """
        self._check_fitted()
        
        if isinstance(texts, str):
            texts = [texts]
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        processed = self.preprocessor.batch_preprocess(texts)
        features = self.extractor.transform(processed)
        
        return self.trainer.predict_proba(features)
    
    def predict_with_confidence(
        self,
        texts: Union[List[str], str]
    ) -> List[Dict[str, Any]]:
        """
        Get predictions with confidence scores.
        
        Args:
            texts: Text(s) to classify
            
        Returns:
            List of dicts with 'label' and 'confidence' keys
        """
        if isinstance(texts, str):
            texts = [texts]
        
        predictions = self.predict(texts)
        probabilities = self.predict_proba(texts)
        
        results = []
        for pred, proba in zip(predictions, probabilities):
            confidence = proba.max()
            results.append({
                'label': pred,
                'confidence': float(confidence),
                'all_probabilities': {
                    str(cls): float(p)
                    for cls, p in zip(self._classes, proba)
                }
            })
        
        return results
    
    def evaluate(
        self,
        texts: Union[List[str], pd.Series],
        labels: Union[List[str], np.ndarray, pd.Series],
        verbose: bool = True
    ) -> ModelMetrics:
        """
        Evaluate classifier performance.
        
        Args:
            texts: Test texts
            labels: True labels
            verbose: Whether to print results
            
        Returns:
            ModelMetrics with evaluation results
        """
        self._check_fitted()
        
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        if isinstance(labels, pd.Series):
            labels = labels.values
        
        processed = self.preprocessor.batch_preprocess(texts)
        features = self.extractor.transform(processed)
        
        metrics = self.trainer.evaluate(features, labels)
        
        if verbose:
            print("\nClassification Results:")
            print(f"  Accuracy:  {metrics.accuracy:.4f}")
            print(f"  Precision: {metrics.precision:.4f}")
            print(f"  Recall:    {metrics.recall:.4f}")
            print(f"  F1 Score:  {metrics.f1:.4f}")
            if metrics.roc_auc is not None:
                print(f"  ROC-AUC:   {metrics.roc_auc:.4f}")
            print("\nDetailed Classification Report:")
            print(metrics.classification_report)
        
        return metrics
    
    def cross_validate(
        self,
        texts: Union[List[str], pd.Series],
        labels: Union[List[str], np.ndarray, pd.Series],
        cv: int = 5,
        scoring: str = 'f1_weighted'
    ) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            texts: Training texts
            labels: Training labels
            cv: Number of folds
            scoring: Scoring metric
            
        Returns:
            Cross-validation scores
        """
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        if isinstance(labels, pd.Series):
            labels = labels.values
        
        processed = self.preprocessor.batch_preprocess(texts)
        features = self.extractor.fit_transform(processed)
        
        return self.trainer.cross_validate(features, labels, cv=cv, scoring=scoring)
    
    def tune_hyperparameters(
        self,
        texts: Union[List[str], pd.Series],
        labels: Union[List[str], np.ndarray, pd.Series],
        param_grid: Optional[Dict] = None,
        cv: int = 5
    ) -> Dict[str, Any]:
        """
        Tune model hyperparameters.
        
        Args:
            texts: Training texts
            labels: Training labels
            param_grid: Parameters to search
            cv: Cross-validation folds
            
        Returns:
            Best parameters and scores
        """
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        if isinstance(labels, pd.Series):
            labels = labels.values
        
        processed = self.preprocessor.batch_preprocess(texts)
        features = self.extractor.fit_transform(processed)
        
        results = self.trainer.tune_hyperparameters(
            features, labels, param_grid=param_grid, cv=cv
        )
        
        self._is_fitted = True
        self._classes = np.unique(labels)
        
        return results
    
    def get_top_features(self, n_top: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get most important features for each class.
        
        Args:
            n_top: Number of top features to return
            
        Returns:
            Dictionary mapping classes to top features
        """
        self._check_fitted()
        
        if self.vectorizer_type == 'hash':
            raise ValueError("Cannot get features from HashingVectorizer")
        
        feature_names = self.extractor.get_feature_names()
        
        if hasattr(self.trainer.model, 'feature_log_prob_'):
            log_probs = self.trainer.model.feature_log_prob_
            top_features = {}
            
            for idx, cls in enumerate(self._classes):
                probs = log_probs[idx]
                top_idx = probs.argsort()[-n_top:][::-1]
                top_features[str(cls)] = [
                    (feature_names[i], float(np.exp(probs[i])))
                    for i in top_idx
                ]
            
            return top_features
        
        return {}
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save the complete classifier to disk.
        
        Args:
            filepath: Path to save the classifier
        """
        self._check_fitted()
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'preprocessor': self.preprocessor,
                'extractor': self.extractor,
                'trainer': self.trainer,
                'classes': self._classes,
                'metadata': self._training_metadata,
                'model_type': self.model_type,
                'vectorizer_type': self.vectorizer_type
            }, f)
        
        print(f"Classifier saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'WordsClassifier':
        """
        Load a saved classifier.
        
        Args:
            filepath: Path to saved classifier
            
        Returns:
            Loaded WordsClassifier instance
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        classifier = cls.__new__(cls)
        classifier.preprocessor = data['preprocessor']
        classifier.extractor = data['extractor']
        classifier.trainer = data['trainer']
        classifier._classes = data['classes']
        classifier._training_metadata = data['metadata']
        classifier.model_type = data['model_type']
        classifier.vectorizer_type = data['vectorizer_type']
        classifier._is_fitted = True
        
        print(f"Classifier loaded from {filepath}")
        return classifier
    
    def _check_fitted(self) -> None:
        """Raise error if classifier is not fitted."""
        if not self._is_fitted:
            raise RuntimeError(
                "Classifier must be fitted before this operation. "
                "Call fit() first."
            )
    
    @property
    def classes(self) -> Optional[np.ndarray]:
        """Get the class labels."""
        return self._classes
    
    @property
    def training_metadata(self) -> Dict:
        """Get metadata from training."""
        return self._training_metadata
    
    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"WordsClassifier(model='{self.model_type}', "
            f"vectorizer='{self.vectorizer_type}', {status})"
        )


def compare_models(
    texts: List[str],
    labels: np.ndarray,
    models: List[str] = None,
    cv: int = 5,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compare multiple classification models.
    
    Args:
        texts: Training texts
        labels: Training labels
        models: List of model types to compare
        cv: Cross-validation folds
        verbose: Whether to print progress
        
    Returns:
        DataFrame with comparison results
    """
    if models is None:
        models = ['naive_bayes', 'svm', 'random_forest', 'logistic']
    
    results = []
    
    for model_type in models:
        if verbose:
            print(f"Evaluating {model_type}...")
        
        classifier = WordsClassifier(model_type=model_type)
        cv_results = classifier.cross_validate(texts, labels, cv=cv)
        
        results.append({
            'model': model_type,
            'mean_f1': cv_results['mean_score'],
            'std_f1': cv_results['std_score']
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values('mean_f1', ascending=False).reset_index(drop=True)
    
    if verbose:
        print("\nModel Comparison Results:")
        print(df.to_string(index=False))
    
    return df
