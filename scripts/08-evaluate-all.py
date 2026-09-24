# ==============================================================================
# Script to evaluate all models in a version and generate comparison figures.
# Generates model comparison bar chart, per-class F1 heatmap, and feature importances.
# ==============================================================================

import argparse
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

from common import get_paths, load_config, load_data

TARGET_CLASSES = ['CAM', 'CB', 'CDM', 'CM', 'GK', 'LB', 'LW', 'RB', 'RW', 'ST']


def evaluate_model(model, X_test, y_test):
    """Compute overall and per-class metrics for a model."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    class_f1s = {cls: report[cls]['f1-score'] for cls in TARGET_CLASSES if cls in report}

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'class_f1s': class_f1s,
    }


def plot_model_comparison(results_df, figures_dir, version):
    """Plot grouped bar chart of overall metrics across all models."""
    plt.figure(figsize=(10, 6))
    melted = results_df.melt(id_vars='Model', value_vars=['Accuracy', 'Precision', 'Recall', 'Macro F1'],
                             var_name='Metric', value_name='Score')

    ax = sns.barplot(data=melted, x='Model', y='Score', hue='Metric', palette='Set2')
    plt.title(f'Model Performance Comparison ({version})', fontsize=14, pad=12)
    plt.ylim(0, 1.05)
    plt.ylabel('Score')
    plt.xlabel('')
    plt.legend(loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=8, xytext=(0, 2),
                        textcoords='offset points')

    plt.tight_layout()
    out_path = os.path.join(figures_dir, f'model_comparison_{version}.png')
    plt.savefig(out_path)
    plt.close()
    print(f'Saved: {out_path}')


def plot_class_f1_heatmap(class_f1_df, figures_dir, version):
    """Plot heatmap showing F1 scores for each position across models."""
    plt.figure(figsize=(9, 7))
    sns.heatmap(class_f1_df, annot=True, fmt='.2f', cmap='YlGnBu', vmin=0, vmax=1.0)
    plt.title(f'Per-Class Macro F1 Score Heatmap ({version})', fontsize=14, pad=12)
    plt.ylabel('Player Position Class')
    plt.xlabel('Model')
    plt.tight_layout()
    out_path = os.path.join(figures_dir, f'per_class_f1_heatmap_{version}.png')
    plt.savefig(out_path)
    plt.close()
    print(f'Saved: {out_path}')


def plot_feature_importance(rf_model, feature_names, figures_dir, version):
    """Plot top 25 feature importances from the Random Forest model."""
    # Extract tree estimator if in pipeline
    if hasattr(rf_model, 'feature_importances_'):
        importances = rf_model.feature_importances_
    elif hasattr(rf_model, 'named_steps'):
        for step in rf_model.named_steps.values():
            if hasattr(step, 'feature_importances_'):
                importances = step.feature_importances_
                break
    else:
        return

    feat_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances,
    }).sort_values('Importance', ascending=False).head(25)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=feat_df, y='Feature', x='Importance', palette='viridis')
    plt.title(f'Top 25 Feature Importances - Random Forest ({version})', fontsize=14, pad=12)
    plt.xlabel('Gini Importance')
    plt.ylabel('')
    plt.tight_layout()
    out_path = os.path.join(figures_dir, f'feature_importance_rf_{version}.png')
    plt.savefig(out_path)
    plt.close()
    print(f'Saved: {out_path}')


def plot_height_weight_distribution(test_df, figures_dir, version):
    """Plot physical distributions across positions if height and weight exist."""
    if 'height_cm' not in test_df.columns or 'weight_kg' not in test_df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.boxplot(data=test_df, x='team_position', y='height_cm', order=TARGET_CLASSES, ax=axes[0], palette='Blues')
    axes[0].set_title('Height (cm) Distribution by Position')
    axes[0].set_xlabel('Position')
    axes[0].set_ylabel('Height (cm)')
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    sns.boxplot(data=test_df, x='team_position', y='weight_kg', order=TARGET_CLASSES, ax=axes[1], palette='Greens')
    axes[1].set_title('Weight (kg) Distribution by Position')
    axes[1].set_xlabel('Position')
    axes[1].set_ylabel('Weight (kg)')
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)

    plt.suptitle(f'Physical Attributes by Position ({version})', fontsize=14)
    plt.tight_layout()
    out_path = os.path.join(figures_dir, f'height_weight_by_position_{version}.png')
    plt.savefig(out_path)
    plt.close()
    print(f'Saved: {out_path}')


def plot_version_comparison():
    """Compare v1 vs v1.1 overall metrics if both version configs exist."""
    v1_cfg_path = 'configs/v1.json'
    v1_1_cfg_path = 'configs/v1.1.json'
    if not (os.path.exists(v1_cfg_path) and os.path.exists(v1_1_cfg_path)):
        return

    # Check if models exist in both
    v1_models_dir = 'models/v1'
    v1_1_models_dir = 'models/v1.1'
    if not os.path.exists(os.path.join(v1_models_dir, 'ensemble_model_v1.pkl')) or \
       not os.path.exists(os.path.join(v1_1_models_dir, 'ensemble_model_v1.1.pkl')):
        return

    v1_cfg = load_config(v1_cfg_path)
    v1_1_cfg = load_config(v1_1_cfg_path)

    _, _, X_test_v1, y_test_v1 = load_data(v1_cfg)
    _, _, X_test_v1_1, y_test_v1_1 = load_data(v1_1_cfg)

    model_names = ['knn', 'logistic_regression', 'random_forest', 'svm', 'ensemble']
    labels = ['KNN', 'Logistic Reg', 'Random Forest', 'SVM', 'Ensemble']

    data = []
    for m_name, label in zip(model_names, labels):
        m1 = joblib.load(os.path.join(v1_models_dir, f'{m_name}_model_v1.pkl'))
        m2 = joblib.load(os.path.join(v1_1_models_dir, f'{m_name}_model_v1.1.pkl'))

        f1_v1 = f1_score(y_test_v1, m1.predict(X_test_v1), average='macro', zero_division=0)
        f1_v1_1 = f1_score(y_test_v1_1, m2.predict(X_test_v1_1), average='macro', zero_division=0)

        data.append({'Model': label, 'Version': 'v1 (51 feats)', 'Macro F1': f1_v1})
        data.append({'Model': label, 'Version': 'v1.1 (77 feats)', 'Macro F1': f1_v1_1})

    comp_df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=comp_df, x='Model', y='Macro F1', hue='Version', palette=['#4c72b0', '#55a868'])
    plt.title('Version Comparison: Macro F1 (v1 vs v1.1)', fontsize=14, pad=12)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f'{h:.2f}', (p.get_x() + p.get_width() / 2., h),
                        ha='center', va='bottom', fontsize=9, xytext=(0, 2),
                        textcoords='offset points')

    plt.tight_layout()
    os.makedirs('figures/v1.1', exist_ok=True)
    out_path = 'figures/v1.1/v1_vs_v1_1_comparison.png'
    plt.savefig(out_path)
    plt.close()
    print(f'Saved: {out_path}')


def main():
    parser = argparse.ArgumentParser(description='Evaluate all models in a version and generate charts.')
    parser.add_argument('--config', type=str, default='configs/v1.json', help='Path to configuration JSON.')
    args = parser.parse_args()

    config = load_config(args.config)
    paths = get_paths(config)
    version = config.get('model_version', 'v1')

    print(f'=== Comprehensive Evaluation ({version}) ===')
    X_train, y_train, X_test, y_test = load_data(config)

    # Dictionary of models to load
    model_definitions = {
        'KNN': f'knn_model_{version}.pkl',
        'Logistic Regression': f'logistic_regression_model_{version}.pkl',
        'Random Forest': f'random_forest_model_{version}.pkl',
        'SVM': f'svm_model_{version}.pkl',
        'Ensemble': f'ensemble_model_{version}.pkl',
    }

    summary_rows = []
    class_f1_dict = {}
    rf_model = None

    for label, filename in model_definitions.items():
        filepath = os.path.join(paths['models_dir'], filename)
        if not os.path.exists(filepath):
            print(f'Warning: Model file not found: {filepath}. Skipping.')
            continue

        model = joblib.load(filepath)
        if label == 'Random Forest':
            rf_model = model

        eval_res = evaluate_model(model, X_test, y_test)
        summary_rows.append({
            'Model': label,
            'Accuracy': eval_res['accuracy'],
            'Precision': eval_res['precision'],
            'Recall': eval_res['recall'],
            'Macro F1': eval_res['f1'],
        })
        class_f1_dict[label] = eval_res['class_f1s']

    if not summary_rows:
        print('No models were found to evaluate.')
        return

    results_df = pd.DataFrame(summary_rows)
    print('\nEvaluation Summary Table:')
    print(results_df.to_string(index=False))

    # Generate charts
    print('\nGenerating visualization suite...')
    plot_model_comparison(results_df, paths['figures_dir'], version)

    class_f1_df = pd.DataFrame(class_f1_dict).reindex(TARGET_CLASSES)
    plot_class_f1_heatmap(class_f1_df, paths['figures_dir'], version)

    if rf_model is not None:
        plot_feature_importance(rf_model, list(X_train.columns), paths['figures_dir'], version)

    test_df_path = os.path.join(paths['processed_dir'], 'test_data.csv')
    test_df = pd.read_csv(test_df_path)
    plot_height_weight_distribution(test_df, paths['figures_dir'], version)

    # Check if we can also plot v1 vs v1.1
    plot_version_comparison()

    print('\nEvaluation and visualizations completed successfully.')


if __name__ == '__main__':
    main()
