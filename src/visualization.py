"""
Visualization Module

Provides plotting utilities for text classification analysis,
including confusion matrices, ROC curves, and feature importance.

Author: Tharun Ponnam
"""

from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)
from wordcloud import WordCloud


def setup_plotting_style():
    """Configure matplotlib style for consistent visualizations."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'figure.dpi': 100,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'font.family': 'sans-serif'
    })


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    normalize: bool = False,
    title: str = 'Confusion Matrix',
    cmap: str = 'Blues',
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot a confusion matrix heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels for axes
        normalize: Whether to normalize by row
        title: Plot title
        cmap: Colormap name
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure object
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = '.2%'
    else:
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=labels or np.unique(y_true),
        yticklabels=labels or np.unique(y_true),
        ax=ax,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
    )
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = 'ROC Curve',
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot ROC curve with AUC score.
    
    Args:
        y_true: True binary labels
        y_proba: Prediction probabilities for positive class
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure object
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(fpr, tpr, color='#2563eb', lw=2,
            label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='#94a3b8', lw=1.5, linestyle='--',
            label='Random Classifier')
    
    ax.fill_between(fpr, tpr, alpha=0.2, color='#2563eb')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = 'Precision-Recall Curve',
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot precision-recall curve.
    
    Args:
        y_true: True binary labels
        y_proba: Prediction probabilities
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure object
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(recall, precision, color='#16a34a', lw=2,
            label=f'PR curve (AUC = {pr_auc:.4f})')
    ax.fill_between(recall, precision, alpha=0.2, color='#16a34a')
    
    baseline = y_true.sum() / len(y_true)
    ax.axhline(y=baseline, color='#94a3b8', linestyle='--',
               label=f'Baseline ({baseline:.4f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_class_distribution(
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = 'Class Distribution',
    figsize: Tuple[int, int] = (8, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot bar chart of class distribution.
    
    Args:
        labels: Array of class labels
        class_names: Names for each class
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure object
    """
    unique, counts = np.unique(labels, return_counts=True)
    
    if class_names is None:
        class_names = [str(c) for c in unique]
    
    colors = ['#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.bar(class_names, counts, color=colors[:len(unique)], 
                  edgecolor='white', linewidth=1.5)
    
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.annotate(f'{count:,}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 5),
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_text_length_distribution(
    texts: List[str],
    labels: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    title: str = 'Text Length Distribution',
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot distribution of text lengths.
    
    Args:
        texts: List of text strings
        labels: Optional class labels for grouping
        class_names: Names for each class
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure object
    """
    lengths = [len(t) for t in texts]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if labels is not None:
        unique_labels = np.unique(labels)
        colors = ['#3b82f6', '#ef4444', '#22c55e', '#f59e0b']
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            label_lengths = [l for l, m in zip(lengths, mask) if m]
            name = class_names[i] if class_names else str(label)
            
            ax.hist(label_lengths, bins=50, alpha=0.6,
                   label=name, color=colors[i % len(colors)])
    else:
        ax.hist(lengths, bins=50, color='#3b82f6', 
                edgecolor='white', alpha=0.8)
    
    ax.set_xlabel('Text Length (characters)')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    
    if labels is not None:
        ax.legend()
    
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_model_comparison(
    results: pd.DataFrame,
    metric: str = 'mean_f1',
    title: str = 'Model Comparison',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot comparison of multiple models.
    
    Args:
        results: DataFrame with model comparison results
        metric: Metric column to plot
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['#3b82f6', '#22c55e', '#f59e0b', '#ef4444', 
              '#8b5cf6', '#06b6d4', '#ec4899']
    
    bars = ax.barh(results['model'], results[metric],
                   color=colors[:len(results)],
                   edgecolor='white', linewidth=1.5)
    
    if 'std_f1' in results.columns:
        ax.errorbar(results[metric], results['model'],
                   xerr=results['std_f1'],
                   fmt='none', color='#374151', capsize=3)
    
    for bar, score in zip(bars, results[metric]):
        ax.text(score + 0.005, bar.get_y() + bar.get_height()/2,
               f'{score:.4f}', va='center', fontsize=10)
    
    ax.set_xlabel(metric.replace('_', ' ').title())
    ax.set_ylabel('Model')
    ax.set_title(title)
    ax.set_xlim(0, 1.1)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_wordcloud(
    texts: List[str],
    title: str = 'Word Cloud',
    max_words: int = 100,
    figsize: Tuple[int, int] = (12, 6),
    colormap: str = 'viridis',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Generate word cloud from texts.
    
    Args:
        texts: List of text strings
        title: Plot title
        max_words: Maximum words in cloud
        figsize: Figure size
        colormap: Matplotlib colormap name
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure object
    """
    combined_text = ' '.join(texts)
    
    wc = WordCloud(
        width=1200,
        height=600,
        max_words=max_words,
        background_color='white',
        colormap=colormap,
        min_font_size=10,
        random_state=42
    ).generate(combined_text)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(wc, interpolation='bilinear')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_feature_importance(
    feature_names: List[str],
    importances: List[float],
    n_top: int = 20,
    title: str = 'Top Feature Importance',
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot top feature importances.
    
    Args:
        feature_names: Names of features
        importances: Importance scores
        n_top: Number of top features to show
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure object
    """
    indices = np.argsort(importances)[-n_top:]
    top_features = [feature_names[i] for i in indices]
    top_importances = [importances[i] for i in indices]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, n_top))
    
    bars = ax.barh(range(n_top), top_importances, color=colors)
    ax.set_yticks(range(n_top))
    ax.set_yticklabels(top_features)
    
    ax.set_xlabel('Importance Score')
    ax.set_ylabel('Feature')
    ax.set_title(title)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_analysis_dashboard(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    texts: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create comprehensive analysis dashboard.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities
        texts: Original text data
        class_names: Names for classes
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure object
    """
    n_plots = 3 if y_proba is not None else 2
    if texts is not None:
        n_plots += 1
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    cm = confusion_matrix(y_true, y_pred)
    labels = class_names or np.unique(y_true)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].set_title('Confusion Matrix')
    
    unique, counts = np.unique(y_true, return_counts=True)
    axes[1].bar(labels, counts, color=['#3b82f6', '#ef4444'])
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Class Distribution')
    
    if y_proba is not None:
        proba = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
        fpr, tpr, _ = roc_curve(y_true, proba)
        axes[2].plot(fpr, tpr, color='#2563eb', lw=2,
                    label=f'AUC = {auc(fpr, tpr):.4f}')
        axes[2].plot([0, 1], [0, 1], '--', color='gray')
        axes[2].set_xlabel('False Positive Rate')
        axes[2].set_ylabel('True Positive Rate')
        axes[2].set_title('ROC Curve')
        axes[2].legend()
    
    if texts is not None:
        lengths = [len(t) for t in texts]
        axes[3].hist(lengths, bins=50, color='#22c55e', alpha=0.8)
        axes[3].set_xlabel('Text Length')
        axes[3].set_ylabel('Frequency')
        axes[3].set_title('Text Length Distribution')
    else:
        axes[3].axis('off')
    
    plt.suptitle('Classification Analysis Dashboard', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
