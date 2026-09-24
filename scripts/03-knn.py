# ==============================================================================
# KNN Classifier for player position classification.
# Supports --tune (hyperparameter search) and --train (fast direct training).
# ==============================================================================

import argparse
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
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
    """Create KNN pipeline with optional hyperparameter configuration."""
    if params is None:
        knn = KNeighborsClassifier()
    else:
        # Strip 'knn__' prefix if present from grid search parameter keys
        clean_params = {k.replace('knn__', ''): v for k, v in params.items()}
        knn = KNeighborsClassifier(**clean_params)

    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('knn', knn),
    ])


def main():
    parser = argparse.ArgumentParser(description='Train or tune KNN classifier.')
    parser.add_argument('--config', type=str, default=None, help='Path to configuration JSON. Defaults to active_version in config.json.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--tune', action='store_true', help='Run GridSearchCV and save best parameters to config.')
    group.add_argument('--train', action='store_true', help='Train directly using saved parameters from config.')
    args = parser.parse_args()

    config = load_config(args.config)
    paths = get_paths(config)
    version = config.get('model_version', 'v1')

    print(f'=== K-Nearest Neighbors ({version}) ===')
    X_train, y_train, X_test, y_test = load_data(config)

    model_file = os.path.join(paths['models_dir'], f'knn_model_{version}.pkl')

    if args.tune:
        print('Running hyperparameter tuning (GridSearchCV)...')
        pipeline = build_pipeline()
        param_grid = {
            'knn__n_neighbors': range(1, 40),
            'knn__weights': ['uniform', 'distance'],
            'knn__metric': ['euclidean', 'manhattan'],
        }
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='recall_macro', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        print('Best parameters found:', best_params)
        print('Best cross-validation score: {:.4f}'.format(grid_search.best_score_))

        # Save winning params back to config file
        save_best_params(paths['config_file'], 'knn', best_params)
        model = grid_search.best_estimator_

        # Optional: Save k-values tuning figure
        plt.figure(figsize=(10, 6))
        mean_scores = grid_search.cv_results_['mean_test_score']
        k_values = grid_search.cv_results_['param_knn__n_neighbors'].data
        plt.plot(k_values, mean_scores, marker='o')
        plt.title(f'KNN ({version}) - F1 Score vs Number of Neighbors (k)')
        plt.xlabel('Number of Neighbors (k)')
        plt.ylabel('Mean Score (Cross-Validation)')
        plt.grid(True)
        tune_fig_path = os.path.join(paths['figures_dir'], f'knn_{version}_f1_score_vs_k.png')
        plt.savefig(tune_fig_path)
        plt.close()
        print(f'Saved tuning curve: {tune_fig_path}')

    else:
        # Direct training mode
        saved_params = config.get('best_params', {}).get('knn')
        if not saved_params:
            raise ValueError(f"No saved hyperparameters found for 'knn' in {args.config}. Run with --tune first.")

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
    save_confusion_matrices(y_test, y_pred, class_labels, paths['figures_dir'], 'KNN', version)


if __name__ == '__main__':
    main()