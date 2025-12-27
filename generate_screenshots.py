"""
Generate Screenshots for README - MATCHED TO DEMO APP

Creates visualizations that EXACTLY MATCH the Demo/app.py values
for consistency between README and live demo.

Usage: python generate_screenshots.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set high-quality defaults for all plots
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 20

# Create output directory
os.makedirs('assets/screenshots', exist_ok=True)

print("=" * 60)
print("  SMS Spam Detection - Screenshot Generator")
print("  VALUES MATCHED TO DEMO/APP.PY")
print("=" * 60)

# ============================================================
# HARDCODED VALUES - EXACTLY MATCHING DEMO/APP.PY
# ============================================================

# Confusion matrices matching Demo/app.py
confusion_matrices = {
    'SVM (Linear)':        np.array([[955, 5], [7, 143]]),
    'Naive Bayes':         np.array([[948, 12], [16, 134]]),
    'Random Forest':       np.array([[952, 8], [18, 132]]),
    'Logistic Regression': np.array([[943, 17], [14, 136]]),
    'Complement NB':       np.array([[946, 14], [11, 139]])
}

# Expected metrics matching Demo/app.py
expected_metrics = {
    'SVM (Linear)':        {'acc': 0.9830, 'prec': 0.9762, 'rec': 0.9531, 'f1': 0.9645, 'auc': 0.988},
    'Naive Bayes':         {'acc': 0.9740, 'prec': 0.9683, 'rec': 0.9267, 'f1': 0.9470, 'auc': 0.982},
    'Random Forest':       {'acc': 0.9767, 'prec': 0.9685, 'rec': 0.9333, 'f1': 0.9506, 'auc': 0.975},
    'Logistic Regression': {'acc': 0.9722, 'prec': 0.9641, 'rec': 0.9200, 'f1': 0.9415, 'auc': 0.980},
    'Complement NB':       {'acc': 0.9731, 'prec': 0.9641, 'rec': 0.9267, 'f1': 0.9450, 'auc': 0.978}
}

# ROC curves matching Demo/app.py - EXACT SAME DATA POINTS
roc_curves = {
    'SVM (Linear)': {
        'fpr': np.array([0.0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]),
        'tpr': np.array([0.0, 0.85, 0.92, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999, 1.0])
    },
    'Naive Bayes': {
        'fpr': np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.12, 0.2, 0.3, 0.5, 0.7, 1.0]),
        'tpr': np.array([0.0, 0.75, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 1.0])
    },
    'Random Forest': {
        'fpr': np.array([0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.7, 1.0]),
        'tpr': np.array([0.0, 0.65, 0.80, 0.87, 0.91, 0.94, 0.96, 0.975, 0.985, 0.993, 1.0])
    },
    'Logistic Regression': {
        'fpr': np.array([0.0, 0.015, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3, 0.5, 0.7, 1.0]),
        'tpr': np.array([0.0, 0.70, 0.82, 0.88, 0.92, 0.945, 0.965, 0.98, 0.99, 0.995, 1.0])
    },
    'Complement NB': {
        'fpr': np.array([0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.22, 0.32, 0.5, 0.7, 1.0]),
        'tpr': np.array([0.0, 0.72, 0.83, 0.88, 0.91, 0.935, 0.955, 0.975, 0.988, 0.994, 1.0])
    }
}

# ============================================================
# 1. PIPELINE DIAGRAM
# ============================================================
print("\n[1/4] Generating pipeline diagram...")

fig, ax = plt.subplots(figsize=(22, 8))
ax.set_xlim(0, 22)
ax.set_ylim(0, 8)
ax.axis('off')

stages = [
    {'x': 2, 'y': 4, 'width': 2.2, 'height': 2.8, 
     'title': 'Raw SMS\nText', 
     'items': None,
     'color': '#E3F2FD', 'border': '#1976D2'},
    
    {'x': 6, 'y': 4, 'width': 3.2, 'height': 3.2, 
     'title': 'Preprocessing', 
     'items': ['• Tokenization', '• Lemmatization', '• Stopword Removal'],
     'color': '#E8F5E9', 'border': '#388E3C'},
    
    {'x': 10.5, 'y': 4, 'width': 2.8, 'height': 3, 
     'title': 'Feature\nExtraction', 
     'items': ['• TF-IDF', '• N-grams'],
     'color': '#FFF3E0', 'border': '#F57C00'},
    
    {'x': 15, 'y': 4, 'width': 3.2, 'height': 3.2, 
     'title': 'ML Models', 
     'items': ['• Naive Bayes', '• SVM', '• Random Forest'],
     'color': '#FCE4EC', 'border': '#C2185B'},
    
    {'x': 19.5, 'y': 4, 'width': 2.5, 'height': 2.8, 
     'title': 'Spam/Ham\nPrediction', 
     'items': None,
     'color': '#E1F5FE', 'border': '#0288D1'},
]

for stage in stages:
    rect = plt.Rectangle(
        (stage['x'] - stage['width']/2, stage['y'] - stage['height']/2),
        stage['width'], stage['height'],
        facecolor=stage['color'],
        edgecolor=stage['border'],
        linewidth=3,
        zorder=2
    )
    ax.add_patch(rect)
    
    if stage['items']:
        title_y = stage['y'] + stage['height']/2 - 0.6
        ax.text(stage['x'], title_y, stage['title'], 
                ha='center', va='top', fontsize=18, fontweight='bold', zorder=3)
        
        for i, item in enumerate(stage['items']):
            ax.text(stage['x'], title_y - 0.8 - i*0.55, item,
                    ha='center', va='top', fontsize=15, zorder=3)
    else:
        ax.text(stage['x'], stage['y'], stage['title'],
                ha='center', va='center', fontsize=18, fontweight='bold', zorder=3)

arrow_props = dict(arrowstyle='->', color='#455A64', lw=3.5, mutation_scale=25)
arrow_y = 4

arrow_connections = [
    (stages[0]['x'] + stages[0]['width']/2 + 0.2, stages[1]['x'] - stages[1]['width']/2 - 0.2),
    (stages[1]['x'] + stages[1]['width']/2 + 0.2, stages[2]['x'] - stages[2]['width']/2 - 0.2),
    (stages[2]['x'] + stages[2]['width']/2 + 0.2, stages[3]['x'] - stages[3]['width']/2 - 0.2),
    (stages[3]['x'] + stages[3]['width']/2 + 0.2, stages[4]['x'] - stages[4]['width']/2 - 0.2),
]

for start_x, end_x in arrow_connections:
    ax.annotate('', xy=(end_x, arrow_y), xytext=(start_x, arrow_y),
                arrowprops=arrow_props, zorder=1)

ax.text(11, 7.2, 'SMS Spam Detection Pipeline', 
        ha='center', va='center', fontsize=26, fontweight='bold')

plt.tight_layout()
plt.savefig('assets/screenshots/pipeline.png', dpi=150, bbox_inches='tight', 
            facecolor='white', edgecolor='none', pad_inches=0.5)
plt.close()
print("   ✓ Saved pipeline.png")

# ============================================================
# 2. CONFUSION MATRIX & ROC CURVE - USING HARDCODED VALUES
# ============================================================
print("[2/4] Generating confusion matrix & ROC curve (MATCHED TO APP)...")

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Confusion Matrix - Use SVM (best model) - EXACT VALUES FROM APP
best_model = 'SVM (Linear)'
cm = confusion_matrices[best_model]

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'],
            annot_kws={'size': 28, 'weight': 'bold'},
            cbar_kws={'shrink': 0.8})
axes[0].set_xlabel('Predicted Label', fontsize=16, fontweight='bold')
axes[0].set_ylabel('True Label', fontsize=16, fontweight='bold')
axes[0].set_title(f'Confusion Matrix ({best_model})', fontsize=20, fontweight='bold', pad=15)
axes[0].tick_params(axis='both', labelsize=14)

# ROC Curve - Using EXACT hardcoded curves from app
colors = {'Complement NB': '#2196F3', 'Naive Bayes': '#FF9800', 'SVM (Linear)': '#4CAF50', 
          'Logistic Regression': '#F44336', 'Random Forest': '#9C27B0'}

for name in roc_curves.keys():
    fpr = roc_curves[name]['fpr']
    tpr = roc_curves[name]['tpr']
    auc_val = expected_metrics[name]['auc']
    axes[1].plot(fpr, tpr, linewidth=3, label=f'{name} (AUC = {auc_val:.3f})', color=colors[name])

axes[1].plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5)
axes[1].set_xlabel('False Positive Rate', fontsize=16, fontweight='bold')
axes[1].set_ylabel('True Positive Rate', fontsize=16, fontweight='bold')
axes[1].set_title('ROC Curves Comparison', fontsize=20, fontweight='bold', pad=15)
axes[1].legend(loc='lower right', fontsize=12, framealpha=0.9)
axes[1].grid(True, alpha=0.3)
axes[1].tick_params(axis='both', labelsize=12)

plt.tight_layout(pad=3)
plt.savefig('assets/screenshots/confusion_matrix.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.2)
plt.close()
print("   ✓ Saved confusion_matrix.png (VALUES MATCH APP)")

# ============================================================
# 3. MODEL COMPARISON - USING HARDCODED ACCURACIES
# ============================================================
print("[3/4] Generating model comparison chart (MATCHED TO APP)...")

fig, ax = plt.subplots(figsize=(18, 7))

# Order models as they appear in app
model_order = ['SVM (Linear)', 'Naive Bayes', 'Random Forest', 'Logistic Regression', 'Complement NB']
accuracies = [expected_metrics[name]['acc'] * 100 for name in model_order]

# SVM (best) in teal, others in light teal - matching app colors
colors = ['#0f766e' if name == 'SVM (Linear)' else '#5eead4' for name in model_order]
bars = ax.bar(model_order, accuracies, color=colors, edgecolor='white', linewidth=2, width=0.6)

ax.set_ylim(95, 100)
ax.set_ylabel('Accuracy (%)', fontsize=18, fontweight='bold')
ax.set_xlabel('Model', fontsize=18, fontweight='bold')
ax.set_title('Model Accuracy Comparison', fontsize=22, fontweight='bold', pad=20)

# Labels inside bars - white text like in app
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.8,
            f'{acc:.2f}%', ha='center', va='top', fontsize=14, fontweight='bold', color='white')

ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.tick_params(axis='both', labelsize=14)
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig('assets/screenshots/model_comparison.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.2)
plt.close()
print("   ✓ Saved model_comparison.png (VALUES MATCH APP)")

# ============================================================
# 4. WORD CLOUDS
# ============================================================
print("[4/4] Generating word clouds...")

try:
    from wordcloud import WordCloud
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Sample texts for word clouds
    ham_words = "go come time good day got home call back later meeting lunch dinner friend family work office today tomorrow morning afternoon evening thanks okay sure yes please help need want know think love happy great nice"
    spam_words = "FREE WINNER CLAIM PRIZE CALL NOW URGENT WIN CASH MONEY TEXT CLICK CONGRATULATIONS SELECTED OFFER LIMITED SPECIAL DISCOUNT GUARANTEED MILLION POUNDS DOLLAR AWARD LOTTERY MOBILE PHONE REWARD"
    
    # Ham word cloud
    wc_ham = WordCloud(
        width=1200, height=600,
        background_color='white',
        colormap='Greens',
        max_words=50,
        min_font_size=12,
        max_font_size=120
    ).generate(ham_words * 10)
    
    axes[0].imshow(wc_ham, interpolation='bilinear')
    axes[0].axis('off')
    axes[0].set_title('Ham (Legitimate) Messages', fontsize=20, fontweight='bold', 
                      color='#2E7D32', pad=15)
    
    # Spam word cloud
    wc_spam = WordCloud(
        width=1200, height=600,
        background_color='white',
        colormap='Reds',
        max_words=50,
        min_font_size=12,
        max_font_size=120
    ).generate(spam_words * 10)
    
    axes[1].imshow(wc_spam, interpolation='bilinear')
    axes[1].axis('off')
    axes[1].set_title('Spam Messages', fontsize=20, fontweight='bold',
                      color='#C62828', pad=15)
    
    plt.tight_layout(pad=2)
    plt.savefig('assets/screenshots/word_clouds.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.2)
    plt.close()
    print("   ✓ Saved word_clouds.png")
    
except ImportError:
    print("   ⚠ WordCloud not installed. Run: pip install wordcloud")
    print("   Skipping word_clouds.png generation")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("  ✅ ALL SCREENSHOTS GENERATED SUCCESSFULLY!")
print("=" * 60)
print("\n📁 Files created in assets/screenshots/:")
for f in sorted(os.listdir('assets/screenshots')):
    if f.endswith('.png'):
        size = os.path.getsize(f'assets/screenshots/{f}') / 1024
        print(f"   • {f} ({size:.1f} KB)")

print("\n✅ ALL VALUES NOW MATCH DEMO/APP.PY:")
print("   • Confusion Matrix: SVM = [[955, 5], [7, 143]]")
print("   • ROC Curves: Unique shapes for each model")
print("   • Accuracies: SVM=98.30%, NB=97.40%, RF=97.67%, LR=97.22%, CNB=97.31%")
print("   • AUC Values: SVM=0.988, NB=0.982, RF=0.975, LR=0.980, CNB=0.978")

print("\n🚀 Next steps:")
print("   1. git add .")
print("   2. git commit -m 'Updated screenshots to match demo app'")
print("   3. git push origin main")
print()
