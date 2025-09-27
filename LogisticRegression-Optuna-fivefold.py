import optuna
import pandas as pd
import numpy as np
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, cohen_kappa_score
from sklearn.linear_model import LogisticRegression

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

# Define the objective function for Logistic Regression
def objective(trial):
    # Suggest hyperparameters from Optuna
    C = trial.suggest_float('C', 0.001, 10, log=True)
    penalty = trial.suggest_categorical('penalty', ['l2', None])
    solver = trial.suggest_categorical('solver', ['lbfgs', 'saga'])

    # Define Logistic Regression model
    model = LogisticRegression(
        C=C,
        penalty=penalty,
        solver=solver,
        max_iter=500,
        random_state=1
    )

    # Perform 5-fold cross-validation and compute mean accuracy
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    mean_cv_score = np.mean(cv_scores)

    # Return 1 - mean accuracy since Optuna minimizes the objective
    return 1 - mean_cv_score

# Optimize hyperparameters with Optuna
sampler = TPESampler(seed=42)
study = optuna.create_study(direction='minimize', sampler=sampler)
study.optimize(objective, n_trials=100, show_progress_bar=True)

# Print best hyperparameters
print("Best Trial:")
trial = study.best_trial
for key, value in trial.params.items():
    print(f"  {key}: {value}")

# Train Logistic Regression with best hyperparameters
best_model = LogisticRegression(
    C=trial.params['C'],
    penalty=trial.params['penalty'],
    solver=trial.params['solver'],
    max_iter=500,
    random_state=1
)

# Fit the best model and compute predictions
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
