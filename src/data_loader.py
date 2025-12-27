"""
Data Loading Module

Utilities for loading and preparing datasets for text classification,
with built-in support for UCI ML Repository datasets.

Author: Tharun Ponnam
"""

import os
import zipfile
import urllib.request
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from io import BytesIO

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class DataLoader:
    """
    Load and prepare datasets for text classification.
    
    Provides methods to load the UCI SMS Spam Collection
    and other text classification datasets.
    """
    
    UCI_SMS_SPAM_URL = (
        "https://archive.ics.uci.edu/static/public/228/"
        "sms+spam+collection.zip"
    )
    
    DATA_DIR = Path(__file__).parent.parent / "data"
    
    @classmethod
    def load_sms_spam(
        cls,
        data_path: Optional[str] = None,
        download: bool = True
    ) -> pd.DataFrame:
        """
        Load the UCI SMS Spam Collection dataset.
        
        Args:
            data_path: Path to existing data file
            download: Whether to download if not found
            
        Returns:
            DataFrame with 'text' and 'label' columns
        """
        if data_path and Path(data_path).exists():
            return cls._load_sms_file(data_path)
        
        default_path = cls.DATA_DIR / "SMSSpamCollection"
        
        if default_path.exists():
            return cls._load_sms_file(str(default_path))
        
        if download:
            return cls._download_sms_spam()
        
        raise FileNotFoundError(
            "SMS Spam dataset not found. Set download=True to fetch it."
        )
    
    @classmethod
    def _download_sms_spam(cls) -> pd.DataFrame:
        """Download SMS Spam dataset from UCI repository."""
        print("Downloading SMS Spam Collection from UCI ML Repository...")
        
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        try:
            with urllib.request.urlopen(cls.UCI_SMS_SPAM_URL, timeout=30) as response:
                zip_data = BytesIO(response.read())
            
            with zipfile.ZipFile(zip_data, 'r') as zf:
                for name in zf.namelist():
                    if 'SMSSpamCollection' in name:
                        with zf.open(name) as f:
                            content = f.read().decode('latin-1')
                        
                        file_path = cls.DATA_DIR / "SMSSpamCollection"
                        with open(file_path, 'w', encoding='utf-8') as out:
                            out.write(content)
                        
                        print(f"Dataset saved to {file_path}")
                        return cls._load_sms_file(str(file_path))
            
        except Exception as e:
            print(f"Download failed: {e}")
            print("Please download manually from:")
            print("  https://archive.ics.uci.edu/dataset/228/sms+spam+collection")
            raise
        
        raise FileNotFoundError("SMSSpamCollection file not found in archive")
    
    @classmethod
    def _load_sms_file(cls, filepath: str) -> pd.DataFrame:
        """Load SMS Spam dataset from file."""
        df = pd.read_csv(
            filepath,
            sep='\t',
            names=['label', 'text'],
            encoding='utf-8'
        )
        
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        
        print(f"Loaded {len(df)} samples")
        print(f"  Ham:  {(df['label'] == 0).sum()}")
        print(f"  Spam: {(df['label'] == 1).sum()}")
        
        return df
    
    @classmethod
    def load_from_csv(
        cls,
        filepath: str,
        text_column: str = 'text',
        label_column: str = 'label',
        encoding: str = 'utf-8'
    ) -> pd.DataFrame:
        """
        Load text classification dataset from CSV file.
        
        Args:
            filepath: Path to CSV file
            text_column: Name of text column
            label_column: Name of label column
            encoding: File encoding
            
        Returns:
            DataFrame with standardized columns
        """
        df = pd.read_csv(filepath, encoding=encoding)
        
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in file")
        if label_column not in df.columns:
            raise ValueError(f"Column '{label_column}' not found in file")
        
        result = pd.DataFrame({
            'text': df[text_column],
            'label': df[label_column]
        })
        
        return result.dropna()
    
    @staticmethod
    def split_data(
        df: pd.DataFrame,
        test_size: float = 0.2,
        validation_size: float = 0.0,
        random_state: int = 42,
        stratify: bool = True
    ) -> Dict[str, Tuple[pd.Series, pd.Series]]:
        """
        Split dataset into train/validation/test sets.
        
        Args:
            df: DataFrame with 'text' and 'label' columns
            test_size: Fraction for test set
            validation_size: Fraction for validation set
            random_state: Random seed for reproducibility
            stratify: Whether to stratify by label
            
        Returns:
            Dictionary with 'train', 'test', and optionally 'val' splits
        """
        stratify_col = df['label'] if stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'],
            df['label'],
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_col
        )
        
        splits = {
            'train': (X_train, y_train),
            'test': (X_test, y_test)
        }
        
        if validation_size > 0:
            val_ratio = validation_size / (1 - test_size)
            stratify_col = y_train if stratify else None
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_train,
                y_train,
                test_size=val_ratio,
                random_state=random_state,
                stratify=stratify_col
            )
            
            splits['train'] = (X_train, y_train)
            splits['val'] = (X_val, y_val)
        
        return splits
    
    @staticmethod
    def get_dataset_stats(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute statistics for a text classification dataset.
        
        Args:
            df: DataFrame with 'text' and 'label' columns
            
        Returns:
            Dictionary with dataset statistics
        """
        text_lengths = df['text'].str.len()
        word_counts = df['text'].str.split().str.len()
        
        label_counts = df['label'].value_counts().to_dict()
        
        return {
            'total_samples': len(df),
            'n_classes': df['label'].nunique(),
            'label_distribution': label_counts,
            'text_length': {
                'mean': text_lengths.mean(),
                'std': text_lengths.std(),
                'min': text_lengths.min(),
                'max': text_lengths.max()
            },
            'word_count': {
                'mean': word_counts.mean(),
                'std': word_counts.std(),
                'min': word_counts.min(),
                'max': word_counts.max()
            }
        }


def load_sample_data() -> pd.DataFrame:
    """
    Load sample data for quick testing.
    
    Returns:
        DataFrame with sample text classification data
    """
    samples = [
        ("Great product, highly recommend!", 1),
        ("Terrible experience, waste of money", 0),
        ("Excellent quality and fast shipping", 1),
        ("Broken on arrival, very disappointed", 0),
        ("Best purchase I've ever made", 1),
        ("Complete garbage, don't buy", 0),
        ("Amazing service, will buy again", 1),
        ("Awful customer support, avoid", 0),
        ("Perfect fit, exactly as described", 1),
        ("Misleading description, poor quality", 0),
    ]
    
    return pd.DataFrame(samples, columns=['text', 'label'])
