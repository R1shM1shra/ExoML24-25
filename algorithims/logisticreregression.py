import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# Load data
train_df = pd.read_csv(r"C:\Users\894054\PycharmProjects\PythonProject\tra1n_split_3.csv")
test_df = pd.read_csv(r"C:\Users\894054\PycharmProjects\PythonProject\t3st_split_3.csv")

# Select features and target
X_train = train_df.iloc[:, 2:]
y_train = train_df.iloc[:, 1]

X_test = test_df.iloc[:, 2:]
y_test = test_df.iloc[:, 1]

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# fit model
logreg = LogisticRegression(random_state=16, max_iter=1000)
logreg.fit(X_train_scaled, y_train)

# Predict
y_pred = logreg.predict(X_test_scaled)

# Evaluate
cnf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cnf_matrix)

# Display
disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix)
disp.plot()

from sklearn.metrics import accuracy_score

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

import numpy as np

# Get feature names
feature_names = X_train.columns

# Get coefficients
coefficients = logreg.coef_[0]

# Create DataFrame for feature importance
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients,
    'Importance (abs)': np.abs(coefficients)
}).sort_values(by='Importance (abs)', ascending=False)

print(importance_df)
