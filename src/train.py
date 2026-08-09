"""
Women's Health Risk Predictor - Training Script
================================================
Trains multiple ML models on the Wisconsin Breast Cancer dataset
and saves the best model to disk.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)
import joblib
import os
import json

# ─── Config ──────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset.csv')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COLS = [
    'clump_thickness', 'size_uniformity', 'shape_uniformity',
    'marginal_adhesion', 'epithelial_size', 'bare_nucleoli',
    'bland_chromatin', 'normal_nucleoli', 'mitoses'
]


# ─── Load & Preprocess ───────────────────────────────────────────────────────
def load_and_preprocess(path: str) -> tuple:
    """Load dataset, handle missing values, encode target."""
    df = pd.read_csv(path)

    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution (raw):\n{df['class'].value_counts()}\n")

    # Replace '?' with NaN, then fill with column median
    df['bare_nucleoli'] = pd.to_numeric(df['bare_nucleoli'], errors='coerce')
    df['bare_nucleoli'] = df['bare_nucleoli'].fillna(df['bare_nucleoli'].median())

    # Drop ID column
    df.drop(columns=['id'], inplace=True)

    # Encode: 2 → 0 (Benign), 4 → 1 (Malignant)
    df['class'] = df['class'].map({2: 0, 4: 1})

    print(f"Missing values after cleaning: {df.isnull().sum().sum()}")
    print(f"Class distribution (encoded): Benign={df['class'].eq(0).sum()}, Malignant={df['class'].eq(1).sum()}\n")

    X = df[FEATURE_COLS].values
    y = df['class'].values
    return X, y, df


# ─── Train Models ─────────────────────────────────────────────────────────────
def train_models(X_train, y_train):
    """Train multiple classifiers and return fitted models."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        'SVM': SVC(probability=True, random_state=RANDOM_STATE),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, random_state=RANDOM_STATE),
    }

    trained = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        trained[name] = model

    return trained, scaler, X_train_scaled


# ─── Evaluate ─────────────────────────────────────────────────────────────────
def evaluate_models(trained, scaler, X_test, y_test) -> pd.DataFrame:
    """Evaluate all models and return a results DataFrame."""
    X_test_scaled = scaler.transform(X_test)
    results = []

    for name, model in trained.items():
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        results.append({
            'Model': name,
            'Accuracy': round(accuracy_score(y_test, y_pred), 4),
            'Precision': round(precision_score(y_test, y_pred), 4),
            'Recall': round(recall_score(y_test, y_pred), 4),
            'F1 Score': round(f1_score(y_test, y_pred), 4),
            'ROC-AUC': round(roc_auc_score(y_test, y_prob), 4),
        })

    df_results = pd.DataFrame(results).sort_values('ROC-AUC', ascending=False)
    return df_results, X_test_scaled


# ─── Save Best Model ──────────────────────────────────────────────────────────
def save_best_model(trained, scaler, df_results, X_test_scaled, y_test):
    """Save the best-performing model and its metadata."""
    best_name = df_results.iloc[0]['Model']
    best_model = trained[best_name]

    # Save model + scaler bundle
    bundle = {'model': best_model, 'scaler': scaler, 'features': FEATURE_COLS}
    model_path = os.path.join(MODEL_DIR, 'breast_cancer_model.pkl')
    joblib.dump(bundle, model_path)
    print(f"\n[OK] Best model saved: {best_name} -> {model_path}")

    # Save LR coefficients as JSON for the web app (if best is LR or for reference)
    lr_model = trained.get('Logistic Regression')
    if lr_model:
        coeffs = {
            'coef': lr_model.coef_[0].tolist(),
            'intercept': float(lr_model.intercept_[0]),
            'scaler_mean': scaler.mean_.tolist(),
            'scaler_scale': scaler.scale_.tolist(),
            'features': FEATURE_COLS,
            'best_model': best_name,
            'best_accuracy': float(df_results.iloc[0]['Accuracy']),
            'best_roc_auc': float(df_results.iloc[0]['ROC-AUC']),
        }
        coeffs_path = os.path.join(MODEL_DIR, 'model_coefficients.json')
        with open(coeffs_path, 'w') as f:
            json.dump(coeffs, f, indent=2)
        print(f"[OK] Model coefficients saved for web app -> {coeffs_path}")

    return best_name, best_model


# ─── Plot Results ─────────────────────────────────────────────────────────────
def plot_results(trained, scaler, X_test_scaled, y_test, df_results, df):
    """Generate and save evaluation plots."""
    plots_dir = os.path.join(MODEL_DIR, '..', 'reports')
    os.makedirs(plots_dir, exist_ok=True)

    sns.set_theme(style='darkgrid', palette='muted')
    plt.rcParams['figure.dpi'] = 100

    # ── 1. Model Comparison Bar Chart ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')

    metrics = ['Accuracy', 'F1 Score', 'ROC-AUC']
    colors = sns.color_palette('viridis', len(df_results))

    bars = axes[0].barh(df_results['Model'], df_results['Accuracy'], color=colors)
    axes[0].set_xlabel('Accuracy')
    axes[0].set_title('Accuracy by Model')
    axes[0].set_xlim(0.85, 1.01)
    for bar, val in zip(bars, df_results['Accuracy']):
        axes[0].text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                     f'{val:.3f}', va='center', fontsize=9)

    bars2 = axes[1].barh(df_results['Model'], df_results['ROC-AUC'], color=colors)
    axes[1].set_xlabel('ROC-AUC')
    axes[1].set_title('ROC-AUC by Model')
    axes[1].set_xlim(0.85, 1.01)
    for bar, val in zip(bars2, df_results['ROC-AUC']):
        axes[1].text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                     f'{val:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'model_comparison.png'), bbox_inches='tight')
    plt.close()

    # ── 2. ROC Curves ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    colors_roc = sns.color_palette('tab10', len(trained))
    for (name, model), color in zip(trained.items(), colors_roc):
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, lw=2, label=f'{name} (AUC={auc:.3f})', color=color)

    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves — All Models', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'roc_curves.png'), bbox_inches='tight')
    plt.close()

    # ── 3. Confusion Matrix (Best Model) ─────────────────────────────────────
    best_name = df_results.iloc[0]['Model']
    best_model = trained[best_name]
    y_pred_best = best_model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred_best)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'],
                ax=ax, linewidths=0.5)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_title(f'Confusion Matrix — {best_name}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'), bbox_inches='tight')
    plt.close()

    # ── 4. Feature Importance (Random Forest) ────────────────────────────────
    rf_model = trained['Random Forest']
    importances = pd.Series(rf_model.feature_importances_, index=FEATURE_COLS).sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    colors_feat = ['#ef4444' if v > importances.median() else '#6366f1' for v in importances]
    importances.plot(kind='barh', ax=ax, color=colors_feat)
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title('Feature Importance — Random Forest', fontsize=13, fontweight='bold')
    ax.axvline(importances.median(), color='gray', linestyle='--', alpha=0.5, label='Median')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'feature_importance.png'), bbox_inches='tight')
    plt.close()

    print(f"\n[OK] Plots saved to {os.path.abspath(plots_dir)}/")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("="*60)
    print("  Women's Health Risk Predictor - Model Training")
    print("="*60 + "\n")

    # Load data
    X, y, df = load_and_preprocess(DATA_PATH)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}\n")

    # Train
    print("Training models...")
    trained, scaler, X_train_scaled = train_models(X_train, y_train)

    # Evaluate
    df_results, X_test_scaled = evaluate_models(trained, scaler, X_test, y_test)

    print("\n[RESULTS] Model Results:")
    print(df_results.to_string(index=False))

    # Save
    best_name, best_model = save_best_model(trained, scaler, df_results, X_test_scaled, y_test)

    # Plot
    plot_results(trained, scaler, X_test_scaled, y_test, df_results, df)

    print("\n" + "="*60)
    print(f"  [DONE] Training complete! Best model: {best_name}")
    print("="*60)


if __name__ == '__main__':
    main()
