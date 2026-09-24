# ==============================================================================
# Logistic Regression for player position classification.
# Supports --tune (hyperparameter search) and --train (fast direct training).
# ==============================================================================

import argparse
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import (
    evaluate_predictions,
    get_paths,
    load_config,
    load_data,
    save_best_params,
    save_confusion_matrices,
)


def build_pipeline(params=None):
    """Create Logistic Regression pipeline with scaling."""
    if params is None:
        lr = LogisticRegression(max_iter=2000)
    else:
        # Strip 'logistic__' prefix if present from grid search parameter keys
        clean_params = {k.replace('logistic__', ''): v for k, v in params.items()}
        if 'max_iter' not in clean_params:
            clean_params['max_iter'] = 2000
        lr = LogisticRegression(**clean_params)

    return Pipeline([
        ('scaler', StandardScaler()),
        ('logistic', lr),
    ])


def main():
    parser = argparse.ArgumentParser(description='Train or tune Logistic Regression classifier.')
    parser.add_argument('--config', type=str, default=None, help='Path to configuration JSON. Defaults to active_version in config.json.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--tune', action='store_true', help='Run GridSearchCV and save best parameters to config.')
    group.add_argument('--train', action='store_true', help='Train directly using saved parameters from config.')
    args = parser.parse_args()

    config = load_config(args.config)
    paths = get_paths(config)
    version = config.get('model_version', 'v1')

    print(f'=== Logistic Regression ({version}) ===')
    X_train, y_train, X_test, y_test = load_data(config)

    model_file = os.path.join(paths['models_dir'], f'logistic_regression_model_{version}.pkl')

    if args.tune:
        print('Running hyperparameter tuning (GridSearchCV)...')
        pipeline = build_pipeline()
        param_grid = {
            # 'logistic__C': [0.1, 1.0, 5, 10, 14, 20],
            'logistic__C': range(14,24),
            'logistic__solver': ['lbfgs'],
        }
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='recall_macro', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        print('Best parameters found:', best_params)
        print('Best cross-validation score: {:.4f}'.format(grid_search.best_score_))

        # Save winning params back to config file
        save_best_params(paths['config_file'], 'logistic_regression', best_params)
        model = grid_search.best_estimator_

    else:
        # Direct training mode
        saved_params = config.get('best_params', {}).get('logistic_regression')
        if not saved_params:
            raise ValueError(f"No saved hyperparameters found for 'logistic_regression' in {args.config}. Run with --tune first.")

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
    save_confusion_matrices(y_test, y_pred, class_labels, paths['figures_dir'], 'Logistic_Regression', version)


if __name__ == '__main__':
    main()