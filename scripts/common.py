# ==============================================================================
# Common helper functions for config management, data loading, and figure saving.
# ==============================================================================

import json
import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def resolve_config_path(config_arg=None):
    """Resolve config file path from command line argument or master config.json."""
    if config_arg:
        return config_arg

    # Check for master config.json at project root
    if os.path.exists('config.json'):
        with open('config.json', 'r') as f:
            master = json.load(f)
            if 'active_config' in master:
                return master['active_config']
            elif 'active_version' in master:
                return f"configs/{master['active_version']}.json"

    # Default fallback
    return 'configs/v1.json'


def load_config(config_path=None):
    """Load configuration dictionary from a JSON file, resolving master default if omitted."""
    resolved_path = resolve_config_path(config_path)
    with open(resolved_path, 'r') as f:
        config = json.load(f)

    # Ensure config tracks its own resolved path
    if 'paths' not in config:
        config['paths'] = {}
    config['paths']['config_file'] = resolved_path

    return config


def save_best_params(config_path, model_key, params):
    """Save tuned hyperparameters back into the config JSON file."""
    config = load_config(config_path)
    if 'best_params' not in config:
        config['best_params'] = {}

    # Convert non-serializable objects (like numpy/tuples) to standard python types
    clean_params = {}
    for k, v in params.items():
        if isinstance(v, tuple):
            clean_params[k] = list(v)
        elif hasattr(v, 'item'):
            clean_params[k] = v.item()
        else:
            clean_params[k] = v

    config['best_params'][model_key] = clean_params

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Updated best parameters for '{model_key}' in {config_path}")


def get_paths(config):
    """Retrieve and create directories specified in the config."""
    paths = config['paths']
    os.makedirs(paths['processed_dir'], exist_ok=True)
    os.makedirs(paths['models_dir'], exist_ok=True)
    os.makedirs(paths['figures_dir'], exist_ok=True)
    return paths


def ensure_preprocessed_data(config):
    """Ensure that train_data.csv and test_data.csv exist for the config, generating them if needed."""
    paths = get_paths(config)
    train_path = os.path.join(paths['processed_dir'], 'train_data.csv')
    test_path = os.path.join(paths['processed_dir'], 'test_data.csv')

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        ver = config.get('model_version', 'current version')
        print(f"Processed dataset missing for {ver} in {paths['processed_dir']}. Running preprocessing automatically...")
        import importlib
        preprocessor = importlib.import_module('02-pre-process-data')
        preprocessor.preprocess_data(config)

    return train_path, test_path


def load_data(config):
    """Load processed training and testing datasets according to config."""
    train_path, test_path = ensure_preprocessed_data(config)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop('team_position', axis=1)
    y_train = train_df['team_position']
    X_test = test_df.drop('team_position', axis=1)
    y_test = test_df['team_position']

    return X_train, y_train, X_test, y_test


def evaluate_predictions(y_test, y_pred):
    """Compute and display standard classification metrics."""
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='macro'),
        'recall': recall_score(y_test, y_pred, average='macro'),
        'f1': f1_score(y_test, y_pred, average='macro'),
    }

    print('Accuracy:  {:.4f}'.format(metrics['accuracy']))
    print('Precision: {:.4f}'.format(metrics['precision']))
    print('Recall:    {:.4f}'.format(metrics['recall']))
    print('Macro F1:  {:.4f}'.format(metrics['f1']))
    print('\nClassification Report:\n', classification_report(y_test, y_pred))

    return metrics


def save_confusion_matrices(y_test, y_pred, class_labels, figures_dir, model_name, version):
    """Save both count and normalized confusion matrix plots."""
    os.makedirs(figures_dir, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred, labels=class_labels)
    cm_norm = confusion_matrix(y_test, y_pred, labels=class_labels, normalize='true')

    # 1. Raw counts confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    plt.title(f'{model_name} ({version}) - Confusion Matrix (Counts)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    raw_path = os.path.join(figures_dir, f'{model_name.lower().replace(" ", "_")}_{version}_confusion_matrix.png')
    plt.savefig(raw_path)
    plt.close()

    # 2. Normalized (recall) confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    plt.title(f'{model_name} ({version}) - Normalized Confusion Matrix (Recall)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    norm_path = os.path.join(figures_dir, f'{model_name.lower().replace(" ", "_")}_{version}_confusion_matrix_normalized.png')
    plt.savefig(norm_path)
    plt.close()

    print(f'Saved figures:\n  - {raw_path}\n  - {norm_path}')


def patch_model_compatibility(model):
    """Ensure backward compatibility for estimators pickled with older scikit-learn versions."""
    if model is None:
        return model

    # In scikit-learn 1.6, SimpleImputer used `_fit_dtype`, whereas 1.9+ requires `_fill_dtype`
    if hasattr(model, '_fit_dtype') and not hasattr(model, '_fill_dtype'):
        model._fill_dtype = model._fit_dtype

    # Recursively patch composite estimators / pipelines / voting ensembles / search objects
    if hasattr(model, 'named_steps'):
        for step in model.named_steps.values():
            patch_model_compatibility(step)
    if hasattr(model, 'steps'):
        for _, step in model.steps:
            patch_model_compatibility(step)
    if hasattr(model, 'estimators_'):
        for est in model.estimators_:
            patch_model_compatibility(est)
    if hasattr(model, 'named_estimators_'):
        for est in model.named_estimators_.values():
            patch_model_compatibility(est)
    if hasattr(model, 'best_estimator_'):
        patch_model_compatibility(model.best_estimator_)
    if hasattr(model, 'final_estimator_'):
        patch_model_compatibility(model.final_estimator_)

    return model


def load_model(filepath):
    """Load a pickled model artifact and patch cross-version compatibility attributes."""
    model = joblib.load(filepath)
    return patch_model_compatibility(model)

