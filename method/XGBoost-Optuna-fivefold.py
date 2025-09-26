import optuna
import pandas as pd
import numpy as np
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, cohen_kappa_score
import xgboost as xgb

# Load dataset
file_path = '405case-no nickson.xlsx'
data = pd.read_excel(file_path)
label_encoder = LabelEncoder()

# Prepare data
X = data[['HR', 'N']]
y = data['Stability']
y_encoded = label_encoder.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=16)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Objective function for XGBoost
def objective(trial):
    # Get hyperparameters from Optuna
    n_estimators = trial.suggest_int('n_estimators', 10, 50)
    max_depth = trial.suggest_int('max_depth', 2, 10)
    learning_rate = trial.suggest_float('learning_rate', 0.001, 0.5, log=True)
    gamma = trial.suggest_float('gamma', 0.0, 1.0)
    min_child_weight = trial.suggest_int('min_child_weight', 1, 5)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)

    # Define XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        gamma=gamma,
        min_child_weight=min_child_weight,
        colsample_bytree=colsample_bytree,
        subsample=subsample,
        random_state=1
    )

    # Compute accuracy using 5-fold cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    mean_cv_score = np.mean(cv_scores)

    # Return objective function value (1 - mean CV score, since Optuna minimizes)
    return 1 - mean_cv_score

# Optimize hyperparameters using Optuna
sampler = TPESampler(seed=42)  # Set random seed
study = optuna.create_study(direction='minimize', sampler=sampler)
study.optimize(objective, n_trials=100, show_progress_bar=True)

# Print best hyperparameters
print("Best Trial:")
trial = study.best_trial
for key, value in trial.params.items():
    print(f"  {key}: {value}")

# Train XGBoost model using best hyperparameters
best_model = xgb.XGBClassifier(
    n_estimators=trial.params['n_estimators'],
    max_depth=trial.params['max_depth'],
    learning_rate=trial.params['learning_rate'],
    gamma=trial.params['gamma'],
    min_child_weight=trial.params['min_child_weight'],
    colsample_bytree=trial.params['colsample_bytree'],
    subsample=trial.params['subsample'],
    random_state=1
)

# Train the best model and compute metrics
best_model.fit(X_train_scaled, y_train)

train_preds = best_model.predict(X_train_scaled)
test_preds = best_model.predict(X_test_scaled)

# Compute training metrics
train_accuracy = accuracy_score(y_train, train_preds)
train_precision = precision_score(y_train, train_preds, average='macro')
train_recall = recall_score(y_train, train_preds, average='macro')
train_f1 = f1_score(y_train, train_preds, average='macro')
train_kappa = cohen_kappa_score(y_train, train_preds)

# Compute testing metrics
test_accuracy = accuracy_score(y_test, test_preds)
test_precision = precision_score(y_test, test_preds, average='macro')
test_recall = recall_score(y_test, test_preds, average='macro')
test_f1 = f1_score(y_test, test_preds, average='macro')
test_kappa = cohen_kappa_score(y_test, test_preds)
confusion_matrix_test = confusion_matrix(y_test, test_preds)

# Print training set metrics
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Training Precision: {train_precision:.4f}")
print(f"Training Recall: {train_recall:.4f}")
print(f"Training F1 Score: {train_f1:.4f}")
print(f"Training Cohen's Kappa: {train_kappa:.4f}")

# Print testing set metrics
print("Test Set Metrics:", confusion_matrix_test)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")
print(f"Test Cohen's Kappa: {test_kappa:.4f}")
print("Test Confusion Matrix:")
print(confusion_matrix_test)
