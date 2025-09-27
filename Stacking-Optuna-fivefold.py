import optuna
from catboost import CatBoostClassifier
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, cohen_kappa_score
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd

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

# Create base models for stacking
model1 = GradientBoostingClassifier(
    n_estimators=33,
    learning_rate=0.0664961169465023,
    max_depth=4,
    min_samples_split=6,
    min_samples_leaf=4,
    subsample=0.7727548367453908,
    random_state=1
)

model2 = CatBoostClassifier(
    learning_rate=0.4959728569177294,
    depth=8,
    l2_leaf_reg=2.457755483883723,
    iterations=25,
    verbose=0,
    random_state=1
)

model3 = RandomForestClassifier(
    n_estimators=46,
    max_depth=6,
    min_samples_split=14,
    min_samples_leaf=1,
    max_features='log2',
    criterion='entropy',
    random_state=1
)

base_models = [
    ('gbdt', model1),
    ('catboost', model2),
    ('rf', model3)
]

# Objective function for optimizing stacking classifier with Optuna
def objective(trial):
    # Get hyperparameters from Optuna
    C = trial.suggest_float('C', 0.01, 10, log=True)
    penalty = trial.suggest_categorical('penalty', ['l2', None])
    solver = trial.suggest_categorical('solver', ['lbfgs', 'saga'])

    # Define the stacking classifier with a Logistic Regression meta-learner
    stacking_clf = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            max_iter=500,
            random_state=1
        )
    )

    # Compute accuracy using 5-fold cross-validation
    cv_scores = cross_val_score(stacking_clf, X_train_scaled, y_train, cv=5, scoring='accuracy')
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

# Train final stacking model with best hyperparameters
final_estimator = LogisticRegression(
    C=trial.params['C'],
    penalty=trial.params['penalty'],
    solver=trial.params['solver'],
    max_iter=500,
    random_state=1
)

best_model = StackingClassifier(estimators=base_models, final_estimator=final_estimator, passthrough=True)

# Train the model
best_model.fit(X_train_scaled, y_train)

# Make predictions
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

# Load engineering validation dataset
file_path2 = 'Nickson.xlsx'
data2 = pd.read_excel(file_path2)

# Prepare features and labels for validation
X2 = data2[['HR', 'N']]
y2 = data2['Stability']
y2_encoded = label_encoder.transform(y2)
X2_scaled = scaler.transform(X2)

# Make predictions on validation dataset
y2_preds = best_model.predict(X2_scaled)

# Compute validation accuracy
test_accuracy2 = accuracy_score(y2_encoded, y2_preds)
print(f"Engineering validation set 1 Accuracy: {test_accuracy2:.4f}")
confusion_matrix_test2 = confusion_matrix(y2_encoded, y2_preds)
print(confusion_matrix_test2)
