import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
try:
    data = pd.read_csv('fraud_transactions.csv')
except FileNotFoundError:
    print("Error: Dataset not found. Please download 'fraud_transactions.csv' from Kaggle and place it in the same folder.")
    exit()
print("First few records:")
print(data.head())

print("\nDataset dimensions:", data.shape)
print("\nFraudulent vs Non-Fraudulent transactions:")
print(data['isFraud'].value_counts())
numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
data_cleaned = data[numeric_features].dropna()

X = data_cleaned.drop(['isFraud', 'isFlaggedFraud'], axis=1)
y = data_cleaned['isFraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

print("\nTraining Logistic Regression Model...")
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)
lr_predictions = log_reg.predict(X_test)

print("\n--- Logistic Regression Performance ---")
print("Accuracy:", accuracy_score(y_test, lr_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, lr_predictions))
print("Classification Report:\n", classification_report(y_test, lr_predictions))

print("\nTraining Random Forest Classifier...")
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
print("\n--- Random Forest Performance ---")
print("Accuracy:", accuracy_score(y_test, rf_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_predictions))
print("Classification Report:\n", classification_report(y_test, rf_predictions))
