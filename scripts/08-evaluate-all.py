# ==============================================================================
# Script to evaluate all models in a version and generate comparison figures.
# Generates model comparison bar chart, per-class F1 heatmap, and feature importances.
# ==============================================================================

import argparse
import os
from typing import cast
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

from common import ensure_preprocessed_data, get_paths, load_config, load_data, load_model

TARGET_CLASSES = ['CAM', 'CB', 'CDM', 'CM', 'GK', 'LB', 'LW', 'RB', 'RW', 'ST']


def evaluate_model(model, X_test, y_test):
    """Compute overall and per-class metrics for a model."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    report = cast(dict[str, dict[str, float]], classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    ))
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
        p = cast(Rectangle, p)
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
    sns.barplot(data=feat_df, y='Feature', x='Importance', hue='Feature', palette='viridis', legend=False)
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

    _, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.boxplot(data=test_df, x='team_position', y='height_cm', order=TARGET_CLASSES, ax=axes[0], hue='team_position', palette='Blues', legend=False)
    axes[0].set_title('Height (cm) Distribution by Position')
    axes[0].set_xlabel('Position')
    axes[0].set_ylabel('Height (cm)')
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    sns.boxplot(data=test_df, x='team_position', y='weight_kg', order=TARGET_CLASSES, ax=axes[1], hue='team_position', palette='Greens', legend=False)
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


def parse_version_key(ver):
    """Parse version string (e.g. 'v1.2') into numeric tuple for descending sorting."""
    import re
    match = re.search(r'v?(\d+)(?:\.(\d+))?', str(ver))
    if match:
        major = int(match.group(1))
        minor = int(match.group(2)) if match.group(2) is not None else 0
        return (major, minor)
    return (0, 0)


def plot_version_comparison(figures_dir):
    """Dynamically compare Macro F1 across all versions that have trained models."""
    import glob

    cfg_files = sorted(glob.glob('configs/*.json'))
    if len(cfg_files) < 2:
        return

    configs = []
    for cfg_file in cfg_files:
        try:
            cfg = load_config(cfg_file)
            if cfg.get('model_version'):
                configs.append(cfg)
        except Exception:
            continue

    # Order versions descending: v1.2, v1.1, v1
    configs.sort(key=lambda c: parse_version_key(c.get('model_version', '')), reverse=True)

    model_keys = ['knn', 'logistic_regression', 'random_forest', 'svm', 'ensemble']
    labels = ['KNN', 'Logistic Reg', 'Random Forest', 'SVM', 'Ensemble']

    data = []
    version_summary = {}

    for cfg in configs:
        ver = cfg.get('model_version')
        m_dir = cfg['paths']['models_dir']

        # Automatically ensure test data is preprocessed if missing
        try:
            ensure_preprocessed_data(cfg)
        except Exception as e:
            print(f'Note: Could not automatically preprocess data for {ver}: {e}')
            continue

        test_path = os.path.join(cfg['paths']['processed_dir'], 'test_data.csv')
        if not os.path.exists(test_path):
            continue

        try:
            test_df = pd.read_csv(filepath_or_buffer=test_path)
            X_test = test_df.drop('team_position', axis=1)
            y_test = test_df['team_position']
        except Exception:
            continue

        ver_scores = {}
        for m_key, lbl in zip(model_keys, labels):
            m_path = os.path.join(m_dir, f'{m_key}_model_{ver}.pkl')
            if os.path.exists(m_path):
                try:
                    m = load_model(m_path)
                    if m is None:
                        continue
                    y_pred = m.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
                    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

                    data.append({
                        'Model': lbl,
                        'Version': ver,
                        'Accuracy': acc,
                        'Precision': prec,
                        'Recall': rec,
                        'Macro F1': f1,
                    })
                    ver_scores[lbl] = f1
                except Exception:
                    pass

        if ver_scores:
            version_summary[ver] = ver_scores

    # Only plot if at least 2 versions have evaluated models
    if len(version_summary) < 2:
        print(f'\nSkipping multi-version comparison: found {len(version_summary)} version(s) with evaluated models (at least 2 required).')
        return

    version_order = list(version_summary.keys())
    print('\nMulti-Version Comparison Matrix (Macro F1):')
    matrix_df = pd.DataFrame(version_summary)
    print(matrix_df.to_string())

    comp_df = pd.DataFrame(data)

    # 1. Macro F1 comparison plot
    plt.figure(figsize=(11, 6))
    ax = sns.barplot(
        data=comp_df,
        x='Model',
        y='Macro F1',
        hue='Version',
        hue_order=version_order,
        palette='Set2',
    )
    plt.title('Version Comparison: Macro F1 Across Models', fontsize=14, pad=12)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for p in ax.patches:
        if not isinstance(p, Rectangle):
            continue
        h = p.get_height()
        if h > 0:
            ax.annotate(f'{h:.2f}', (p.get_x() + p.get_width() / 2., h),
                        ha='center', va='bottom', fontsize=9, xytext=(0, 2),
                        textcoords='offset points')

    plt.tight_layout()
    out_path = os.path.join(figures_dir, 'version_comparison_all.png')
    plt.savefig(out_path)
    alt_f1_path = os.path.join(figures_dir, 'version_comparison_macro_f1.png')
    plt.savefig(alt_f1_path)
    # Also save a copy to top-level figures/ directory if figures_dir is a subfolder
    if os.path.isdir('figures') and os.path.abspath(figures_dir) != os.path.abspath('figures'):
        plt.savefig(os.path.join('figures', 'version_comparison_all.png'))
        plt.savefig(os.path.join('figures', 'version_comparison_macro_f1.png'))
    plt.close()
    print(f'Saved: {out_path}')
    print(f'Saved: {alt_f1_path}')

    # 2. Multi-metric comparison plot (Accuracy, Precision, Recall, Macro F1)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    metrics_map = [
        ('Accuracy', 'Accuracy Across Models', axes[0, 0]),
        ('Precision', 'Macro Precision Across Models', axes[0, 1]),
        ('Recall', 'Macro Recall Across Models', axes[1, 0]),
        ('Macro F1', 'Macro F1 Across Models', axes[1, 1]),
    ]

    for metric_col, title, ax_m in metrics_map:
        sns.barplot(
            data=comp_df,
            x='Model',
            y=metric_col,
            hue='Version',
            hue_order=version_order,
            palette='Set2',
            ax=ax_m,
        )
        ax_m.set_title(title, fontsize=13, pad=10)
        ax_m.set_ylim(0, 1.08)
        ax_m.set_ylabel(metric_col)
        ax_m.set_xlabel('')
        ax_m.grid(axis='y', linestyle='--', alpha=0.7)
        for p in ax_m.patches:
            if not isinstance(p, Rectangle):
                continue
            h = p.get_height()
            if h > 0:
                ax_m.annotate(f'{h:.2f}', (p.get_x() + p.get_width() / 2., h),
                              ha='center', va='bottom', fontsize=8, xytext=(0, 2),
                              textcoords='offset points')

    # Single unified legend for the 2x2 grid
    handles, labels_list = axes[0, 0].get_legend_handles_labels()
    for ax_m in axes.flat:
        if ax_m.get_legend() is not None:
            ax_m.get_legend().remove()
    fig.legend(handles, labels_list, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=len(version_order), fontsize=11, title='Model Version')
    fig.suptitle('Version Comparison: All Metrics Across Models', fontsize=16, y=1.02)
    plt.tight_layout()

    out_metrics_path = os.path.join(figures_dir, 'version_comparison_all_metrics.png')
    plt.savefig(out_metrics_path, bbox_inches='tight')
    if os.path.isdir('figures') and os.path.abspath(figures_dir) != os.path.abspath('figures'):
        plt.savefig(os.path.join('figures', 'version_comparison_all_metrics.png'), bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_metrics_path}')



def main():
    parser = argparse.ArgumentParser(description='Evaluate all models in a version and generate charts.')
    parser.add_argument('--config', type=str, default=None, help='Path to configuration JSON. Defaults to active_version in config.json.')
    args = parser.parse_args()

    config = load_config(args.config)
    paths = get_paths(config)
    version = config.get('model_version', 'v1')

    print(f'=== Comprehensive Evaluation ({version}) ===')
    X_train, _, X_test, y_test = load_data(config)

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

        model = load_model(filepath)
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

    # Check if we can also plot multi-version comparison
    plot_version_comparison(paths['figures_dir'])

    print('\nEvaluation and visualizations completed successfully.')


if __name__ == '__main__':
    main()
