# Data Processing
import pandas as pd
import numpy as np

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz

# Load data
train_df = pd.read_csv(r"C:\Users\894054\PycharmProjects\PythonProject\tra1n_split_2.csv")
test_df = pd.read_csv(r"C:\Users\894054\PycharmProjects\PythonProject\t3st_split_2.csv")

# Select features and target (2nd column)
X_train = train_df.iloc[:, 2:]
y_train = train_df.iloc[:, 1]

X_test = test_df.iloc[:, 2:]
y_test = test_df.iloc[:, 1]

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Feature Importance (display numerically)
importances = rf.feature_importances_
feature_names = X_train.columns

# Combine into a DataFrame for cleaner display
feature_importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# Sort by importance descending
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Print the feature importances
print("\nFeature Importances:")
print(feature_importances)