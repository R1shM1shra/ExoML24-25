# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix

# Load data
train_df = pd.read_csv(r"C:\Users\894054\PycharmProjects\PythonProject\tra1n_split_3.csv")
test_df = pd.read_csv(r"C:\Users\894054\PycharmProjects\PythonProject\t3st_split_3.csv")

# Select features and target (2nd column)
X_train = train_df.iloc[:, 2:]
y_train = train_df.iloc[:, 1]

X_test = test_df.iloc[:, 2:]
y_test = test_df.iloc[:, 1]

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus

# Extract feature names dynamically from the dataset
feature_cols = X_train.columns.tolist()

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,
                feature_names=feature_cols,
                class_names=['1', '2'])

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('exoplanets.png')
Image(graph.create_png())

from IPython.display import display
image = Image(filename='exoplanets.png')
display(image)

#  matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

import numpy as np

# Get feature importance
feature_importance = clf.feature_importances_

# Sort features by importance
sorted_idx = np.argsort(feature_importance)[::-1]  # Sort in descending order

print("Feature Importance Ranking:")
for i in sorted_idx:
    print(f"{feature_cols[i]}: {feature_importance[i]:.4f}")