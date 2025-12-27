"""
SMS Spam Detection - Machine Learning-based Spam Classifier

A production-ready SMS spam detection system using NLP and classical machine
learning algorithms. Classifies text messages as spam or ham (legitimate)
with 98%+ accuracy.

Author: Tharun Ponnam
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Tharun Ponnam"
__email__ = "tharunponnam007@gmail.com"

from .classifier import WordsClassifier
from .preprocessing import TextPreprocessor
from .feature_extraction import FeatureExtractor
from .model_trainer import ModelTrainer

__all__ = [
    "WordsClassifier",
    "TextPreprocessor",
    "FeatureExtractor",
    "ModelTrainer",
]
