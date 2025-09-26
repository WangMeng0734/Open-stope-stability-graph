import optuna
import pandas as pd
import numpy as np
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, cohen_kappa_score
from sklearn.tree import DecisionTreeClassifier

# Load dataset
file_path = '405case-no nickson.xlsx'
data = pd.read_excel(file_path)
label_encoder = LabelEncoder()

# Prepare features and labels
X = data[['HR', 'N']]
y = data['Stability']
y_encoded = label_encoder.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=16)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Objective function for Decision Tree
def objective(trial):
    # Suggest hyperparameters from Optuna
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
    max_depth = trial.suggest_int('max_depth', 1, 10)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    max_features = trial.suggest_int('max_features', 1, 2)

    # Define Decision Tree model
    model = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=1
    )

    # Perform 5-fold cross-validation and compute accuracy
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    mean_cv_score = np.mean(cv_scores)

    # Return 1 - mean accuracy since Optuna minimizes the objective
    return 1 - mean_cv_score

# Run hyperparameter optimization with Optuna
sampler = TPESampler(seed=42)
study = optuna.create_study(direction='minimize', sampler=sampler)
study.optimize(objective, n_trials=100, show_progress_bar=True)

# Print the best hyperparameters
print("Best Trial:")
trial = study.best_trial
for key, value in trial.params.items():
    print(f"  {key}: {value}")

# Train Decision Tree with the best hyperparameters
best_model = DecisionTreeClassifier(
    criterion=trial.params['criterion'],
    max_depth=trial.params['max_depth'],
    min_samples_split=trial.params['min_samples_split'],
    min_samples_leaf=trial.params['min_samples_leaf'],
    max_features=trial.params['max_features'],
    random_state=1
)

# Fit the best model
best_model.fit(X_train_scaled, y_train)

# Predictions on training and testing sets
train_preds = best_model.predict(X_train_scaled)
test_preds = best_model.predict(X_test_scaled)

# Training set metrics
train_accuracy = accuracy_score(y_train, train_preds)
train_precision = precision_score(y_train, train_preds, average='macro')
train_recall = recall_score(y_train, train_preds, average='macro')
train_f1 = f1_score(y_train, train_preds, average='macro')
train_kappa = cohen_kappa_score(y_train, train_preds)

# Testing set metrics
test_accuracy = accuracy_score(y_test, test_preds)
test_precision = precision_score(y_test, test_preds, average='macro')
test_recall = recall_score(y_test, test_preds, average='macro')
test_f1 = f1_score(y_test, test_preds, average='macro')
test_kappa = cohen_kappa_score(y_test, test_preds)
confusion_matrix_test = confusion_matrix(y_test, test_preds)

# Print training set metrics
print(f"  Accuracy: {train_accuracy:.4f}")
print(f"  Precision: {train_precision:.4f}")
print(f"  Recall: {train_recall:.4f}")
print(f"  F1 Score: {train_f1:.4f}")
print(f"  Cohen's Kappa: {train_kappa:.4f}")

# Print testing set metrics
print("Test Set Metrics:", confusion_matrix_test)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")
print(f"Test Cohen's Kappa: {test_kappa:.4f}")
print("Test Confusion Matrix:")
print(confusion_matrix_test)
