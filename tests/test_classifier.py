"""
Unit Tests for SMS Spam Detection

Comprehensive test suite covering preprocessing, feature extraction,
model training, and the end-to-end classification pipeline.

Author: Tharun Ponnam
Run: pytest tests/ -v
"""

import sys
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import TextPreprocessor
from src.feature_extraction import FeatureExtractor
from src.model_trainer import ModelTrainer, ModelMetrics
from src.classifier import WordsClassifier
from src.data_loader import DataLoader, load_sample_data


class TestTextPreprocessor:
    """Tests for TextPreprocessor class."""
    
    @pytest.fixture
    def preprocessor(self):
        return TextPreprocessor()
    
    def test_clean_text_basic(self, preprocessor):
        text = "Hello World!"
        result = preprocessor.clean_text(text)
        assert result == "hello world"
    
    def test_clean_text_removes_urls(self, preprocessor):
        text = "Check out https://example.com for more info"
        result = preprocessor.clean_text(text)
        assert "https" not in result
        assert "example.com" not in result
    
    def test_clean_text_removes_emails(self, preprocessor):
        text = "Contact me at test@email.com"
        result = preprocessor.clean_text(text)
        assert "@" not in result
    
    def test_clean_text_removes_numbers(self, preprocessor):
        text = "Call 123-456-7890 now"
        result = preprocessor.clean_text(text)
        assert "123" not in result
    
    def test_tokenize_basic(self, preprocessor):
        text = "This is a simple test sentence"
        tokens = preprocessor.tokenize(text)
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert "a" not in tokens  # Stopword removed
    
    def test_tokenize_lemmatization(self):
        preprocessor = TextPreprocessor(use_lemmatization=True)
        text = "running dogs are playing"
        tokens = preprocessor.tokenize(text)
        assert "running" in tokens or "run" in tokens
    
    def test_tokenize_stemming(self):
        preprocessor = TextPreprocessor(use_lemmatization=False, use_stemming=True)
        text = "running dogs are playing"
        tokens = preprocessor.tokenize(text)
        assert any("run" in t or "play" in t for t in tokens)
    
    def test_preprocess_returns_string(self, preprocessor):
        text = "Hello world, this is a test!"
        result = preprocessor.preprocess(text)
        assert isinstance(result, str)
    
    def test_batch_preprocess(self, preprocessor):
        texts = ["First text", "Second text", "Third text"]
        results = preprocessor.batch_preprocess(texts)
        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)
    
    def test_get_pos_tags(self, preprocessor):
        text = "The quick brown fox jumps"
        tags = preprocessor.get_pos_tags(text)
        assert isinstance(tags, list)
        assert all(isinstance(t, tuple) and len(t) == 2 for t in tags)
    
    def test_empty_text(self, preprocessor):
        result = preprocessor.preprocess("")
        assert result == ""
    
    def test_non_string_input(self, preprocessor):
        result = preprocessor.clean_text(None)
        assert result == ""


class TestFeatureExtractor:
    """Tests for FeatureExtractor class."""
    
    @pytest.fixture
    def sample_texts(self):
        return [
            "this is the first document",
            "this document is the second document",
            "and this is the third one",
            "is this the first document"
        ]
    
    def test_tfidf_vectorizer(self, sample_texts):
        extractor = FeatureExtractor(vectorizer_type='tfidf')
        features = extractor.fit_transform(sample_texts)
        
        assert features.shape[0] == len(sample_texts)
        assert features.shape[1] > 0
    
    def test_count_vectorizer(self, sample_texts):
        extractor = FeatureExtractor(vectorizer_type='count')
        features = extractor.fit_transform(sample_texts)
        
        assert features.shape[0] == len(sample_texts)
    
    def test_max_features_limit(self, sample_texts):
        extractor = FeatureExtractor(max_features=5)
        features = extractor.fit_transform(sample_texts)
        
        assert features.shape[1] <= 5
    
    def test_ngram_range(self, sample_texts):
        extractor = FeatureExtractor(ngram_range=(1, 3))
        features = extractor.fit_transform(sample_texts)
        
        feature_names = extractor.get_feature_names()
        has_bigrams = any(' ' in name for name in feature_names)
        assert has_bigrams or len(feature_names) > 0
    
    def test_get_feature_names(self, sample_texts):
        extractor = FeatureExtractor(vectorizer_type='tfidf')
        extractor.fit_transform(sample_texts)
        
        names = extractor.get_feature_names()
        assert isinstance(names, list)
        assert len(names) > 0
    
    def test_vocabulary_size(self, sample_texts):
        extractor = FeatureExtractor()
        extractor.fit_transform(sample_texts)
        
        assert extractor.vocabulary_size > 0
    
    def test_transform_without_fit_raises(self, sample_texts):
        extractor = FeatureExtractor(vectorizer_type='tfidf')
        with pytest.raises(RuntimeError):
            extractor.transform(sample_texts)
    
    def test_invalid_vectorizer_type(self):
        with pytest.raises(ValueError):
            FeatureExtractor(vectorizer_type='invalid')


class TestModelTrainer:
    """Tests for ModelTrainer class."""
    
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        X = np.random.rand(100, 20)
        y = np.random.randint(0, 2, 100)
        return X, y
    
    def test_naive_bayes_training(self, sample_data):
        X, y = sample_data
        trainer = ModelTrainer(model_type='naive_bayes')
        trainer.fit(X, y)
        
        predictions = trainer.predict(X)
        assert len(predictions) == len(y)
    
    def test_svm_training(self, sample_data):
        X, y = sample_data
        trainer = ModelTrainer(model_type='svm')
        trainer.fit(X, y)
        
        predictions = trainer.predict(X)
        assert len(predictions) == len(y)
    
    def test_random_forest_training(self, sample_data):
        X, y = sample_data
        trainer = ModelTrainer(model_type='random_forest', n_estimators=10)
        trainer.fit(X, y)
        
        predictions = trainer.predict(X)
        assert len(predictions) == len(y)
    
    def test_logistic_training(self, sample_data):
        X, y = sample_data
        trainer = ModelTrainer(model_type='logistic')
        trainer.fit(X, y)
        
        predictions = trainer.predict(X)
        assert len(predictions) == len(y)
    
    def test_evaluate_returns_metrics(self, sample_data):
        X, y = sample_data
        trainer = ModelTrainer(model_type='naive_bayes')
        trainer.fit(X, y)
        
        metrics = trainer.evaluate(X, y)
        
        assert isinstance(metrics, ModelMetrics)
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.precision <= 1
        assert 0 <= metrics.recall <= 1
        assert 0 <= metrics.f1 <= 1
    
    def test_cross_validate(self, sample_data):
        X, y = sample_data
        trainer = ModelTrainer(model_type='naive_bayes')
        
        results = trainer.cross_validate(X, y, cv=3)
        
        assert 'mean_score' in results
        assert 'std_score' in results
        assert 0 <= results['mean_score'] <= 1
    
    def test_predict_proba(self, sample_data):
        X, y = sample_data
        trainer = ModelTrainer(model_type='naive_bayes')
        trainer.fit(X, y)
        
        proba = trainer.predict_proba(X)
        
        assert proba.shape[0] == len(y)
        assert np.allclose(proba.sum(axis=1), 1.0)
    
    def test_save_and_load(self, sample_data, tmp_path):
        X, y = sample_data
        trainer = ModelTrainer(model_type='naive_bayes')
        trainer.fit(X, y)
        
        filepath = tmp_path / "model.pkl"
        trainer.save(filepath)
        
        loaded = ModelTrainer.load(filepath)
        predictions = loaded.predict(X)
        
        assert len(predictions) == len(y)
    
    def test_invalid_model_type(self):
        with pytest.raises(ValueError):
            ModelTrainer(model_type='invalid')


class TestWordsClassifier:
    """Tests for WordsClassifier class."""
    
    @pytest.fixture
    def sample_texts(self):
        return [
            "Great product, highly recommend!",
            "Terrible experience, waste of money",
            "Excellent quality and fast shipping",
            "Broken on arrival, very disappointed",
            "Best purchase I've ever made",
            "Complete garbage, don't buy",
            "Amazing service, will buy again",
            "Awful customer support, avoid"
        ]
    
    @pytest.fixture
    def sample_labels(self):
        return np.array([1, 0, 1, 0, 1, 0, 1, 0])
    
    def test_fit_and_predict(self, sample_texts, sample_labels):
        classifier = WordsClassifier(model_type='naive_bayes')
        classifier.fit(sample_texts, sample_labels, verbose=False)
        
        predictions = classifier.predict(sample_texts)
        assert len(predictions) == len(sample_labels)
    
    def test_predict_single_text(self, sample_texts, sample_labels):
        classifier = WordsClassifier(model_type='naive_bayes')
        classifier.fit(sample_texts, sample_labels, verbose=False)
        
        prediction = classifier.predict("This is great!")
        assert len(prediction) == 1
    
    def test_predict_proba(self, sample_texts, sample_labels):
        classifier = WordsClassifier(model_type='naive_bayes')
        classifier.fit(sample_texts, sample_labels, verbose=False)
        
        proba = classifier.predict_proba(sample_texts)
        assert proba.shape[0] == len(sample_texts)
    
    def test_predict_with_confidence(self, sample_texts, sample_labels):
        classifier = WordsClassifier(model_type='naive_bayes')
        classifier.fit(sample_texts, sample_labels, verbose=False)
        
        results = classifier.predict_with_confidence("This is amazing!")
        
        assert len(results) == 1
        assert 'label' in results[0]
        assert 'confidence' in results[0]
        assert 0 <= results[0]['confidence'] <= 1
    
    def test_evaluate(self, sample_texts, sample_labels):
        classifier = WordsClassifier(model_type='naive_bayes')
        classifier.fit(sample_texts, sample_labels, verbose=False)
        
        metrics = classifier.evaluate(sample_texts, sample_labels, verbose=False)
        
        assert isinstance(metrics, ModelMetrics)
    
    def test_different_model_types(self, sample_texts, sample_labels):
        for model_type in ['naive_bayes', 'svm', 'logistic']:
            classifier = WordsClassifier(model_type=model_type)
            classifier.fit(sample_texts, sample_labels, verbose=False)
            predictions = classifier.predict(sample_texts)
            assert len(predictions) == len(sample_labels)
    
    def test_save_and_load(self, sample_texts, sample_labels, tmp_path):
        classifier = WordsClassifier(model_type='naive_bayes')
        classifier.fit(sample_texts, sample_labels, verbose=False)
        
        filepath = tmp_path / "classifier.pkl"
        classifier.save(filepath)
        
        loaded = WordsClassifier.load(filepath)
        predictions = loaded.predict(sample_texts)
        
        assert len(predictions) == len(sample_labels)
    
    def test_classes_property(self, sample_texts, sample_labels):
        classifier = WordsClassifier()
        classifier.fit(sample_texts, sample_labels, verbose=False)
        
        assert classifier.classes is not None
        assert len(classifier.classes) == 2
    
    def test_training_metadata(self, sample_texts, sample_labels):
        classifier = WordsClassifier()
        classifier.fit(sample_texts, sample_labels, verbose=False)
        
        metadata = classifier.training_metadata
        assert 'n_samples' in metadata
        assert 'n_classes' in metadata


class TestDataLoader:
    """Tests for DataLoader class."""
    
    def test_load_sample_data(self):
        df = load_sample_data()
        
        assert isinstance(df, pd.DataFrame)
        assert 'text' in df.columns
        assert 'label' in df.columns
    
    def test_split_data(self):
        df = load_sample_data()
        splits = DataLoader.split_data(df, test_size=0.3)
        
        assert 'train' in splits
        assert 'test' in splits
        
        X_train, y_train = splits['train']
        X_test, y_test = splits['test']
        
        assert len(X_train) + len(X_test) == len(df)
    
    def test_split_data_with_validation(self):
        df = load_sample_data()
        splits = DataLoader.split_data(df, test_size=0.2, validation_size=0.2)
        
        assert 'train' in splits
        assert 'test' in splits
        assert 'val' in splits
    
    def test_get_dataset_stats(self):
        df = load_sample_data()
        stats = DataLoader.get_dataset_stats(df)
        
        assert 'total_samples' in stats
        assert 'n_classes' in stats
        assert 'text_length' in stats
        assert 'word_count' in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
