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

# SVM Regressor from Scratch
class SVMRegressor:
    def __init__(self, learning_rate=0.01, epochs=1000, C=1.0):
        """
        Initialize a Support Vector Machine Regressor.

        Parameters:
        - learning_rate (float): Learning rate for gradient descent.
        - epochs (int): Number of training epochs.
        - C (float): Regularization parameter.
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.C = C

    def fit(self, X, y):
        """
        Train the SVM Regressor.

        Parameters:
        - X (numpy.ndarray): Training feature data.
        - y (numpy.ndarray): Training target data.
        """
        # Add bias term to X
        X = np.column_stack((np.ones(len(X)), X))
        m, n = X.shape

        # Initialize weights
        self.w = np.zeros(n)

        # Stochastic Gradient Descent
        for epoch in range(self.epochs):
            for i in range(m):
                error = y[i] - np.dot(self.w, X[i])
                if error != 0:
                    self.w = self.w + self.learning_rate * (self.C * error * X[i] - self.w)
                else:
                    self.w = self.w - self.learning_rate * self.w

    def predict(self, X):
        """
        Make predictions using the trained model.

        Parameters:
        - X (numpy.ndarray): Test feature data.

        Returns:
        - numpy.ndarray: Predicted target values.
        """
        # Add bias term to X
        X = np.column_stack((np.ones(len(X)), X))
        return np.dot(X, self.w)

# Instantiate the SVM Regressor
svm_regressor = SVMRegressor(learning_rate=0.01, epochs=1000, C=1.0)

# Train the model
svm_regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred_reg = svm_regressor.predict(X_test)

# Visualize the results for regression
plt.scatter(X_test, y_test, color='black', label='True values')
plt.scatter(X_test, y_pred_reg, color='red', label='Predicted values')
plt.plot(X_test, y_pred_reg, color='blue', linewidth=3, label='Regression Line')
plt.title('SVM Regression from Scratch')
plt.xlabel('Median Income')
plt.ylabel('Median House Value')
plt.legend()
plt.show()
