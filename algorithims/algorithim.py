from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd

# Load data
train_df = pd.read_csv(r"C:\Users\894054\Downloads\datasplitsML\tra1n_split_1.csv")
test_df = pd.read_csv(r"C:\Users\894054\Downloads\datasplitsML\t3st_split_1.csv")

# Select features and target (2nd column)
X_train = train_df.iloc[:, 2:]
y_train = train_df.iloc[:, 1]

X_test = test_df.iloc[:, 2:]
y_test = test_df.iloc[:, 1]

# Normalization using a MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform training data
X_test_scaled = scaler.transform(X_test)        # Transform test data using the same scaler

# Train model
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=40)
svm_model.fit(X_train_scaled, y_train)

# Predictions
y_pred = svm_model.predict(X_test_scaled)

# accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

#  matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
