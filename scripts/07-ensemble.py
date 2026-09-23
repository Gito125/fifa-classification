# ==========================================================================
# This script implements an Ensemble classifier for classification tasks.
# It includes functions to train the model, make predictions, and evaluate its performance.
# ===========================================================================

# Import necessary libraries for Ensemble implementation
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score

# Load the pre-processed testing dataset
test_df = pd.read_csv('data/processed/test_data.csv')

# Separate features and target (team_position) variable for testing dataset
X_test = test_df.drop('team_position', axis=1)
y_test = test_df['team_position']

# Load the saved models from the models directory
model_paths = {
    'KNN': './models/v1/knn_model_v1.pkl',
    'Logistic Regression': './models/v1/logistic_regression_model_v1.pkl',
    'Random Forest': './models/v1/random_forest_model_v1.pkl',
    'SVM': './models/v1/svm_model_v1.pkl',
}
saved_models = {
    model_name: joblib.load(model_path)
    for model_name, model_path in model_paths.items()
}

# Generate predictions with each saved model
model_predictions = {
    model_name: model.predict(X_test)
    for model_name, model in saved_models.items()
}

# Combine the saved model predictions using hard voting
prediction_matrix = np.array(list(model_predictions.values()))
ensemble_predictions = pd.DataFrame(prediction_matrix).mode(axis=0).iloc[0].to_numpy()

# Evaluate each saved model and the hard-voting ensemble on the test set
print('=============================================================\nSaved Model Results:')
for model_name, y_pred in model_predictions.items():
    print(f'\n{model_name}:')
    print('Accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))
    print('Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='macro', zero_division=0)))
    print('Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='macro', zero_division=0)))
    print('F1 Score: {:.2f}'.format(f1_score(y_test, y_pred, average='macro', zero_division=0)))
    print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
    print('Classification Report:\n', classification_report(y_test, y_pred, zero_division=0))

print('\nHard-Voting Ensemble:')
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, ensemble_predictions)))
print('Precision: {:.2f}'.format(precision_score(y_test, ensemble_predictions, average='macro', zero_division=0)))
print('Recall: {:.2f}'.format(recall_score(y_test, ensemble_predictions, average='macro', zero_division=0)))
print('F1 Score: {:.2f}'.format(f1_score(y_test, ensemble_predictions, average='macro', zero_division=0)))
print('Confusion Matrix:\n', confusion_matrix(y_test, ensemble_predictions))
print('Classification Report:\n', classification_report(y_test, ensemble_predictions, zero_division=0))

# Graphical representation of the ensemble results (optional)
print('=============================================================\nGraphical representation of the results:')
plt.figure(figsize=(10, 6))
# Plotting the confusion matrix
sns.heatmap(
    confusion_matrix(y_test, ensemble_predictions),
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=sorted(y_test.unique()),
    yticklabels=sorted(y_test.unique()),
)
plt.title('Hard-Voting Ensemble Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/v1/ensemble_v1_confusion_matrix.png')
plt.show()