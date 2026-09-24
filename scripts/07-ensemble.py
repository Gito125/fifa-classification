# ==============================================================================
# Soft-Voting Ensemble Classifier combining KNN, Logistic Regression, Random Forest, and SVM.
# Loads tuned base estimator parameters directly from version config.
# Supports --tune (weights search) and --train (fast direct training).
# ==============================================================================

import argparse
import os
import joblib
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
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


def build_base_estimators(config):
    """Instantiate base models using hyperparameters stored in config."""
    best_params = config.get('best_params', {})

    # 1. KNN
    knn_params = best_params.get('knn', {'knn__n_neighbors': 4, 'knn__weights': 'uniform', 'knn__metric': 'manhattan'})
    clean_knn = {k.replace('knn__', ''): v for k, v in knn_params.items()}
    knn = KNeighborsClassifier(**clean_knn)

    # 2. Logistic Regression
    lr_params = best_params.get('logistic_regression', {'logistic__C': 14, 'logistic__solver': 'lbfgs', 'logistic__max_iter': 2000})
    clean_lr = {k.replace('logistic__', ''): v for k, v in lr_params.items()}
    if 'max_iter' not in clean_lr:
        clean_lr['max_iter'] = 2000
    logistic = LogisticRegression(**clean_lr)

    # 3. Random Forest
    rf_params = best_params.get('random_forest', {'random_forest__n_estimators': 900, 'random_forest__max_depth': None, 'random_forest__min_samples_leaf': 1})
    clean_rf = {k.replace('random_forest__', ''): v for k, v in rf_params.items()}
    if 'random_state' not in clean_rf:
        clean_rf['random_state'] = 42
    if 'n_jobs' not in clean_rf:
        clean_rf['n_jobs'] = -1
    random_forest = RandomForestClassifier(**clean_rf)

    # 4. SVM
    svm_params = best_params.get('svm', {'svm__C': 17, 'svm__gamma': 'scale', 'svm__kernel': 'rbf'})
    clean_svm = {k.replace('svm__', ''): v for k, v in svm_params.items()}
    clean_svm['probability'] = True
    svm = SVC(**clean_svm)

    estimators = [
        ('knn', knn),
        ('logistic', logistic),
        ('random_forest', random_forest),
        ('svm', svm),
    ]
    return estimators


def build_pipeline(config, weights=None):
    """Construct pipeline containing standard scaler and soft-voting ensemble."""
    estimators = build_base_estimators(config)
    ensemble = VotingClassifier(
        estimators=estimators,
        voting='soft',
        weights=weights,
    )
    return Pipeline([
        ('scaler', StandardScaler()),
        ('ensemble', ensemble),
    ])


def main():
    parser = argparse.ArgumentParser(description='Train or tune Soft-Voting Ensemble classifier.')
    parser.add_argument('--config', type=str, default='configs/v1.json', help='Path to configuration JSON.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--tune', action='store_true', help='Search for optimal voting weights and save to config.')
    group.add_argument('--train', action='store_true', help='Train directly using saved weights from config.')
    args = parser.parse_args()

    config = load_config(args.config)
    paths = get_paths(config)
    version = config.get('model_version', 'v1')

    print(f'=== Soft-Voting Ensemble ({version}) ===')
    X_train, y_train, X_test, y_test = load_data(config)

    model_file = os.path.join(paths['models_dir'], f'ensemble_model_{version}.pkl')

    if args.tune:
        print('Building pipeline with tuned base estimators...')
        pipeline = build_pipeline(config)
        param_grid = {
            'ensemble__weights': [
                (1, 1, 1, 1),
                (2, 1, 1, 1),
                (1, 2, 1, 1),
                (1, 1, 2, 1),
                (1, 1, 1, 2),
                (1, 2, 1, 2),
            ],
        }
        print('Tuning voting weights (GridSearchCV)...')
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        print('Best parameters found:', best_params)
        print('Best cross-validation score: {:.4f}'.format(grid_search.best_score_))

        # Save winning weights back to config file
        save_best_params(args.config, 'ensemble', best_params)
        model = grid_search.best_estimator_

    else:
        # Direct training mode
        ensemble_cfg = config.get('best_params', {}).get('ensemble', {})
        weights = ensemble_cfg.get('ensemble__weights')
        if weights is not None and isinstance(weights, list):
            weights = tuple(weights)

        print(f'Training directly with loaded weights: {weights}')
        model = build_pipeline(config, weights=weights)
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
    save_confusion_matrices(y_test, y_pred, class_labels, paths['figures_dir'], 'Ensemble', version)


if __name__ == '__main__':
    main()