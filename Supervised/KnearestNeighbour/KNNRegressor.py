import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load the housing dataset from CSV file
df = pd.read_csv('housing.csv')

# Extract features and target variable
X = df[['median_income']].values
y = df['median_house_value'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN Regressor
class KNNRegressor:
    def __init__(self, k=3):
        """
        Initialize KNNRegressor.

        Parameters:
        - k (int): Number of neighbors to consider for prediction.
        """
        self.k = k

    def fit(self, X_train, y_train):
        """
        Fit the model with training data.

        Parameters:
        - X_train (numpy.ndarray): Training feature data.
        - y_train (numpy.ndarray): Training target data.
        """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        """
        Predict target values for the given test data.

        Parameters:
        - X_test (numpy.ndarray): Test feature data.

        Returns:
        - numpy.ndarray: Predicted target values.
        """
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        """
        Predict target value for a single data point.

        Parameters:
        - x (numpy.ndarray): Single data point.

        Returns:
        - float: Predicted target value.
        """
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        k_neighbors_indices = np.argsort(distances)[:self.k]
        k_neighbor_labels = [self.y_train[i] for i in k_neighbors_indices]
        predicted_value = np.mean(k_neighbor_labels)
        return predicted_value

# Instantiate the KNN Regressor
knn_regressor = KNNRegressor(k=3)

# Train the model and print predictions on each epoch
epochs = 10
for epoch in range(epochs):
    knn_regressor.fit(X_train, y_train)
    y_pred = knn_regressor.predict(X_test)
    
    print(f"Epoch {epoch + 1}/{epochs} - Predictions: {y_pred}")

# Visualize the results
plt.scatter(X_test, y_test, color='black', label='True values')
plt.scatter(X_test, y_pred, color='red', label='Final Predictions')
plt.title('KNN Regression')
plt.xlabel('Median Income')
plt.ylabel('Median House Value')
plt.legend()
plt.show()
