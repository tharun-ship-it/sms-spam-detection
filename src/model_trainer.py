"""
Model Training Module

Implements multiple machine learning classifiers including
Naive Bayes, SVM, Random Forest, and Logistic Regression
with hyperparameter tuning capabilities.

Author: Tharun Ponnam
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float] = None
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'roc_auc': self.roc_auc
        }


class ModelTrainer:
    """
    Train and evaluate multiple classification models.
    
    Provides a unified interface for training Naive Bayes, SVM,
    Random Forest, and other classifiers with optional
    hyperparameter optimization.
    
    Attributes:
        model_type (str): Type of classifier to use
        model: The underlying sklearn model instance
    """
    
    AVAILABLE_MODELS = {
        'naive_bayes': MultinomialNB,
        'complement_nb': ComplementNB,
        'svm': LinearSVC,
        'svm_rbf': SVC,
        'random_forest': RandomForestClassifier,
        'logistic': LogisticRegression,
        'gradient_boosting': GradientBoostingClassifier,
        'sgd': SGDClassifier,
        'knn': KNeighborsClassifier
    }
    
    DEFAULT_PARAMS = {
        'naive_bayes': {'alpha': 1.0},
        'complement_nb': {'alpha': 1.0},
        'svm': {
            'C': 1.0,
            'max_iter': 2000,
            'random_state': 42
        },
        'svm_rbf': {
            'C': 1.0,
            'kernel': 'rbf',
            'probability': True,
            'random_state': 42
        },
        'random_forest': {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'random_state': 42,
            'n_jobs': -1
        },
        'logistic': {
            'C': 1.0,
            'max_iter': 1000,
            'random_state': 42,
            'n_jobs': -1
        },
        'gradient_boosting': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'random_state': 42
        },
        'sgd': {
            'loss': 'modified_huber',
            'max_iter': 1000,
            'random_state': 42,
            'n_jobs': -1
        },
        'knn': {
            'n_neighbors': 5,
            'weights': 'distance',
            'n_jobs': -1
        }
    }
    
    PARAM_GRIDS = {
        'naive_bayes': {
            'alpha': [0.01, 0.1, 0.5, 1.0, 2.0]
        },
        'complement_nb': {
            'alpha': [0.01, 0.1, 0.5, 1.0, 2.0]
        },
        'svm': {
            'C': [0.1, 1.0, 10.0],
            'loss': ['hinge', 'squared_hinge']
        },
        'random_forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        },
        'logistic': {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
    }
    
    def __init__(
        self,
        model_type: str = 'naive_bayes',
        **kwargs
    ):
        """
        Initialize the model trainer.
        
        Args:
            model_type: Type of classifier to use
            **kwargs: Additional parameters for the model
        """
        if model_type not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"model_type must be one of {list(self.AVAILABLE_MODELS.keys())}"
            )
        
        self.model_type = model_type
        
        params = self.DEFAULT_PARAMS.get(model_type, {}).copy()
        params.update(kwargs)
        
        model_class = self.AVAILABLE_MODELS[model_type]
        self.model = model_class(**params)
        
        self._is_fitted = False
        self._best_params = None
        self._cv_results = None
    
    def fit(
        self,
        X: Union[np.ndarray, csr_matrix],
        y: np.ndarray
    ) -> 'ModelTrainer':
        """
        Train the model on given data.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            Self for method chaining
        """
        self.model.fit(X, y)
        self._is_fitted = True
        return self
    
    def predict(
        self,
        X: Union[np.ndarray, csr_matrix]
    ) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels
        """
        self._check_fitted()
        return self.model.predict(X)
    
    def predict_proba(
        self,
        X: Union[np.ndarray, csr_matrix]
    ) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability array of shape (n_samples, n_classes)
        """
        self._check_fitted()
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        elif hasattr(self.model, 'decision_function'):
            decisions = self.model.decision_function(X)
            if decisions.ndim == 1:
                decisions = decisions.reshape(-1, 1)
                proba = 1 / (1 + np.exp(-decisions))
                return np.hstack([1 - proba, proba])
            return decisions
        raise NotImplementedError(
            f"{self.model_type} does not support probability predictions"
        )
    
    def evaluate(
        self,
        X: Union[np.ndarray, csr_matrix],
        y: np.ndarray,
        average: str = 'weighted'
    ) -> ModelMetrics:
        """
        Evaluate model performance on test data.
        
        Args:
            X: Feature matrix
            y: True labels
            average: Averaging method for metrics
            
        Returns:
            ModelMetrics dataclass with evaluation results
        """
        self._check_fitted()
        
        y_pred = self.predict(X)
        
        roc_auc = None
        try:
            y_proba = self.predict_proba(X)
            if len(np.unique(y)) == 2:
                roc_auc = roc_auc_score(y, y_proba[:, 1])
            else:
                roc_auc = roc_auc_score(y, y_proba, multi_class='ovr')
        except Exception:
            pass
        
        return ModelMetrics(
            accuracy=accuracy_score(y, y_pred),
            precision=precision_score(y, y_pred, average=average, zero_division=0),
            recall=recall_score(y, y_pred, average=average, zero_division=0),
            f1=f1_score(y, y_pred, average=average, zero_division=0),
            roc_auc=roc_auc,
            confusion_matrix=confusion_matrix(y, y_pred),
            classification_report=classification_report(y, y_pred, zero_division=0)
        )
    
    def cross_validate(
        self,
        X: Union[np.ndarray, csr_matrix],
        y: np.ndarray,
        cv: int = 5,
        scoring: str = 'f1_weighted'
    ) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            X: Feature matrix
            y: Target labels
            cv: Number of folds
            scoring: Scoring metric
            
        Returns:
            Dictionary with mean and std scores
        """
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(self.model, X, y, cv=skf, scoring=scoring)
        
        return {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores.tolist()
        }
    
    def tune_hyperparameters(
        self,
        X: Union[np.ndarray, csr_matrix],
        y: np.ndarray,
        param_grid: Optional[Dict] = None,
        cv: int = 5,
        scoring: str = 'f1_weighted',
        n_iter: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Tune model hyperparameters using grid or random search.
        
        Args:
            X: Feature matrix
            y: Target labels
            param_grid: Parameter grid to search
            cv: Number of CV folds
            scoring: Scoring metric
            n_iter: Number of iterations for random search
            
        Returns:
            Dictionary with best parameters and scores
        """
        if param_grid is None:
            param_grid = self.PARAM_GRIDS.get(self.model_type, {})
        
        if not param_grid:
            raise ValueError(f"No parameter grid available for {self.model_type}")
        
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        if n_iter:
            search = RandomizedSearchCV(
                self.model,
                param_grid,
                n_iter=n_iter,
                cv=skf,
                scoring=scoring,
                n_jobs=-1,
                random_state=42
            )
        else:
            search = GridSearchCV(
                self.model,
                param_grid,
                cv=skf,
                scoring=scoring,
                n_jobs=-1
            )
        
        search.fit(X, y)
        
        self.model = search.best_estimator_
        self._is_fitted = True
        self._best_params = search.best_params_
        self._cv_results = search.cv_results_
        
        return {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'cv_results': {
                'mean_test_score': search.cv_results_['mean_test_score'].tolist(),
                'std_test_score': search.cv_results_['std_test_score'].tolist()
            }
        }
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        self._check_fitted()
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'model_type': self.model_type,
                'best_params': self._best_params
            }, f)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'ModelTrainer':
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded ModelTrainer instance
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        trainer = cls.__new__(cls)
        trainer.model = data['model']
        trainer.model_type = data['model_type']
        trainer._is_fitted = True
        trainer._best_params = data.get('best_params')
        trainer._cv_results = None
        
        return trainer
    
    def _check_fitted(self) -> None:
        """Raise error if model is not fitted."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before this operation")
    
    @property
    def best_params(self) -> Optional[Dict]:
        """Get best parameters from hyperparameter tuning."""
        return self._best_params
    
    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return f"ModelTrainer(type='{self.model_type}', {status})"


class EnsembleTrainer:
    """
    Train an ensemble of multiple classifiers.
    
    Combines predictions from multiple models using voting
    or stacking strategies.
    """
    
    def __init__(
        self,
        models: List[Tuple[str, str]],
        voting: str = 'soft'
    ):
        """
        Initialize ensemble trainer.
        
        Args:
            models: List of (name, model_type) tuples
            voting: Voting strategy ('hard' or 'soft')
        """
        self.trainers = []
        estimators = []
        
        for name, model_type in models:
            trainer = ModelTrainer(model_type)
            self.trainers.append((name, trainer))
            estimators.append((name, trainer.model))
        
        self.ensemble = VotingClassifier(
            estimators=estimators,
            voting=voting,
            n_jobs=-1
        )
        self._is_fitted = False
    
    def fit(
        self,
        X: Union[np.ndarray, csr_matrix],
        y: np.ndarray
    ) -> 'EnsembleTrainer':
        """Train all models in the ensemble."""
        self.ensemble.fit(X, y)
        self._is_fitted = True
        return self
    
    def predict(self, X: Union[np.ndarray, csr_matrix]) -> np.ndarray:
        """Make ensemble predictions."""
        return self.ensemble.predict(X)
    
    def evaluate(
        self,
        X: Union[np.ndarray, csr_matrix],
        y: np.ndarray
    ) -> ModelMetrics:
        """Evaluate ensemble performance."""
        y_pred = self.predict(X)
        
        return ModelMetrics(
            accuracy=accuracy_score(y, y_pred),
            precision=precision_score(y, y_pred, average='weighted', zero_division=0),
            recall=recall_score(y, y_pred, average='weighted', zero_division=0),
            f1=f1_score(y, y_pred, average='weighted', zero_division=0),
            confusion_matrix=confusion_matrix(y, y_pred),
            classification_report=classification_report(y, y_pred, zero_division=0)
        )
