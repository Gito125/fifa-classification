# ==============================================================================
# Random Forest Classifier for player position classification.
# Supports --tune (hyperparameter search) and --train (fast direct training).
# ==============================================================================

import argparse
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from common import (
    evaluate_predictions,
    get_paths,
    load_config,
    load_data,
    save_best_params,
    save_confusion_matrices,
)


def build_model(params=None):
    """Instantiate RandomForestClassifier with optional parameters."""
    if params is None:
        return RandomForestClassifier(random_state=42, n_jobs=-1)

    # Strip 'random_forest__' prefix if present from grid search parameter keys
    clean_params = {k.replace('random_forest__', ''): v for k, v in params.items()}
    if 'random_state' not in clean_params:
        clean_params['random_state'] = 42
    if 'n_jobs' not in clean_params:
        clean_params['n_jobs'] = -1

    return RandomForestClassifier(**clean_params)


def main():
    parser = argparse.ArgumentParser(description='Train or tune Random Forest classifier.')
    parser.add_argument('--config', type=str, default='configs/v1.json', help='Path to configuration JSON.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--tune', action='store_true', help='Run GridSearchCV and save best parameters to config.')
    group.add_argument('--train', action='store_true', help='Train directly using saved parameters from config.')
    args = parser.parse_args()

    config = load_config(args.config)
    paths = get_paths(config)
    version = config.get('model_version', 'v1')

    print(f'=== Random Forest ({version}) ===')
    X_train, y_train, X_test, y_test = load_data(config)

    model_file = os.path.join(paths['models_dir'], f'random_forest_model_{version}.pkl')

    if args.tune:
        print('Running hyperparameter tuning (GridSearchCV)...')
        base_rf = build_model()
        param_grid = {
            'n_estimators': [500, 700, 900],
            'max_depth': [None, 30],
            'min_samples_leaf': [1, 2],
        }
        grid_search = GridSearchCV(base_rf, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        print('Best parameters found:', best_params)
        print('Best cross-validation score: {:.4f}'.format(grid_search.best_score_))

        # Save winning params back to config file
        save_best_params(args.config, 'random_forest', best_params)
        model = grid_search.best_estimator_

    else:
        # Direct training mode
        saved_params = config.get('best_params', {}).get('random_forest')
        if not saved_params:
            raise ValueError(f"No saved hyperparameters found for 'random_forest' in {args.config}. Run with --tune first.")

        print(f'Training directly with parameters: {saved_params}')
        model = build_model(saved_params)
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
    save_confusion_matrices(y_test, y_pred, class_labels, paths['figures_dir'], 'Random_Forest', version)


if __name__ == '__main__':
    main()