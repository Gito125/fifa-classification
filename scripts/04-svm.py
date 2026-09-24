# ==============================================================================
# Support Vector Machine (SVM) for player position classification.
# Supports --tune (hyperparameter search) and --train (fast direct training).
# ==============================================================================

import argparse
import os
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import (
    evaluate_predictions,
    get_paths,
    load_config,
    load_data,
    save_best_params,
    save_confusion_matrices,
)


def build_pipeline(params=None):
    """Create SVM pipeline with scaling and probability estimation."""
    if params is None:
        svm = SVC(probability=True)
    else:
        # Strip 'svm__' prefix if present from grid search parameter keys
        clean_params = {k.replace('svm__', ''): v for k, v in params.items()}
        # Ensure probability is always True for ensemble voting compatibility
        clean_params['probability'] = True
        svm = SVC(**clean_params)

    return Pipeline([
        ('scaler', StandardScaler()),
        ('svm', svm),
    ])


def main():
    parser = argparse.ArgumentParser(description='Train or tune SVM classifier.')
    parser.add_argument('--config', type=str, default=None, help='Path to configuration JSON. Defaults to active_version in config.json.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--tune', action='store_true', help='Run GridSearchCV and save best parameters to config.')
    group.add_argument('--train', action='store_true', help='Train directly using saved parameters from config.')
    args = parser.parse_args()

    config = load_config(args.config)
    paths = get_paths(config)
    version = config.get('model_version', 'v1')

    print(f'=== Support Vector Machine ({version}) ===')
    X_train, y_train, X_test, y_test = load_data(config)

    model_file = os.path.join(paths['models_dir'], f'svm_model_{version}.pkl')

    if args.tune:
        print('Running hyperparameter tuning (GridSearchCV)...')
        pipeline = build_pipeline()
        param_grid = {
            'svm__C': [18, 22, 24, *range(25, 29)],  # Expanded range for tuning
            'svm__kernel': ['rbf'],
            'svm__gamma': ['scale', 'auto'],
        }
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='recall_macro', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        print('Best parameters found:', best_params)
        print('Best cross-validation score: {:.4f}'.format(grid_search.best_score_))

        # Save winning params back to config file
        save_best_params(paths['config_file'], 'svm', best_params)
        model = grid_search.best_estimator_

    else:
        # Direct training mode
        saved_params = config.get('best_params', {}).get('svm')
        if not saved_params:
            raise ValueError(f"No saved hyperparameters found for 'svm' in {args.config}. Run with --tune first.")

        print(f'Training directly with parameters: {saved_params}')
        model = build_pipeline(saved_params)
        model.fit(X_train, y_train)

    # Save fitted model artifact
    joblib.dump(model, model_file)
    print(f'Saved model artifact: {model_file}')

    # Evaluate on held-out test set
    print('\nEvaluating model on test dataset:')
    y_pred = model.predict(X_test)
    evaluate_predictions(y_test, y_pred)

    # Save confusion matrices
    class_labels = list(model.classes_)
    save_confusion_matrices(y_test, y_pred, class_labels, paths['figures_dir'], 'SVM', version)


if __name__ == '__main__':
    main()