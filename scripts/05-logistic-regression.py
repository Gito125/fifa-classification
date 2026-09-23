# ==========================================================================
# This script implements Logistic Regression for classification tasks.
# It includes functions to train the model, make predictions, and evaluate its performance.
# ===========================================================================

# Import necessary libraries for Logistic Regression implementation
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Load the pre-processed training and testing datasets
train_df = pd.read_csv('data/processed/train_data.csv')
test_df = pd.read_csv('data/processed/test_data.csv')

# Separate features and target (team_position) variable for training and testing datasets
X_train = train_df.drop('team_position', axis=1)
y_train = train_df['team_position']
X_test = test_df.drop('team_position', axis=1)
y_test = test_df['team_position']

# Create a pipeline that includes data scaling and Logistic Regression classifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logistic', LogisticRegression(max_iter=2000)),
])

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'logistic__C': [0.01, 0.1, 1, 10],
    'logistic__solver': ['lbfgs'],
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)

# Fit the grid search on the training data
grid_search.fit(X_train, y_train)

# Get the best model from the grid search
best_model = grid_search.best_estimator_

print('=============================================================\nGrid Search Results:')
print('Best parameters found: ', grid_search.best_params_)
print('Best cross-validation F1 score: {:.2f}'.format(grid_search.best_score_))

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test)
print('=============================================================\nEvaluating the best model on the test set:')
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))
print('Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='macro')))
print('Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='macro')))
print('F1 Score: {:.2f}'.format(f1_score(y_test, y_pred, average='macro')))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))

# Save the best model to a file for future use
os.makedirs('models', exist_ok=True)
joblib.dump(best_model, 'models/logistic_regression_model_v1.pkl')

# Graphical representation of the results (optional)
print('=============================================================\nGraphical representation of the results:')
plt.figure(figsize=(10, 6))
# Plotting the confusion matrix
sns.heatmap(
    confusion_matrix(y_test, y_pred),
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=[str(label) for label in best_model.classes_],
    yticklabels=[str(label) for label in best_model.classes_],
)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/logistic_regression_v1_confusion_matrix.png')
plt.show()