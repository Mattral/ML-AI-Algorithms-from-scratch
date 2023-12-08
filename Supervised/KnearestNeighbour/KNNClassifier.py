import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the housing dataset from CSV file
df = pd.read_csv('housing.csv')

# Create a binary target variable based on the median
median_value = df['median_house_value'].median()
df['above_median'] = (df['median_house_value'] > median_value).astype(int)

# Extract features and target variable
X = df[['median_income']].values
y = df['above_median'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN Classifier
class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        k_neighbors_indices = np.argsort(distances)[:self.k]
        k_neighbor_labels = [self.y_train[i] for i in k_neighbors_indices]
        predicted_label = max(set(k_neighbor_labels), key=k_neighbor_labels.count)
        return predicted_label

# Instantiate the KNN Classifier
knn_classifier = KNNClassifier(k=3)

# Train the model
knn_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn_classifier.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Visualize the results
plt.scatter(X_test, y_test, color='black', label='True values')
plt.scatter(X_test, y_pred, color='red', label='Predicted values')
plt.title('KNN Binary Classification')
plt.xlabel('Median Income')
plt.ylabel('Above Median House Value (1) / Below Median (0)')
plt.legend()
plt.show()
