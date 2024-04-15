import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class LassoRegression:
    """
    Lasso Regression implementation using coordinate descent.

    Attributes:
        alpha (float): Regularization strength; must be a positive float.
        max_iterations (int): Maximum number of iterations to run the algorithm.
        tol (float): Tolerance for stopping criteria.
        theta (np.array): Coefficients of the regression model.
    """
    def __init__(self, alpha=1.0, max_iterations=1000, tol=1e-4):
        """
        Initializes the LassoRegression model with specified parameters.

        Args:
            alpha (float): Regularization strength; must be a positive float.
            max_iterations (int): Maximum number of iterations to run the algorithm.
            tol (float): Tolerance for stopping criteria.
        """
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.tol = tol
        self.theta = None

    def soft_threshold(self, x, threshold):
        """
        Apply the soft thresholding function used in Lasso for updating the coefficients.

        Args:
            x (float): The coefficient value.
            threshold (float): The threshold value derived from alpha.

        Returns:
            float: The thresholded value.
        """
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

    def fit(self, X, y):
        """
        Fit the Lasso Regression model using coordinate descent.

        Args:
            X (np.array): Feature matrix for training data.
            y (np.array): Target vector for training data.
        """
        # Add a bias term to X
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]

        # Initialize coefficients
        self.theta = np.zeros(X_bias.shape[1])

        for _ in range(self.max_iterations):
            old_theta = self.theta.copy()

            # Coordinate Descent for LASSO Regression
            for j in range(X_bias.shape[1]):
                X_j = X_bias[:, j]
                y_pred = X_bias @ self.theta - X_j * self.theta[j]
                rho = X_j.T @ (y - y_pred)
                self.theta[j] = self.soft_threshold(rho, self.alpha / (X_j.T @ X_j))

            # Check for convergence
            if np.linalg.norm(self.theta - old_theta, 1) < self.tol:
                break

    def predict(self, X):
        """
        Make predictions using the Lasso Regression model.

        Args:
            X (np.array): Feature matrix for prediction.

        Returns:
            np.array: Predicted values.
        """
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        return X_bias @ self.theta

# Load the dataset from housing.csv
df = pd.read_csv('housing.csv')

# Selecting features and target
X = df[['median_income']]
y = df['median_house_value']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and fit the Lasso Regression model
lasso = LassoRegression(alpha=1.0)
lasso.fit(X_train_scaled, y_train)

# Predictions
y_pred_train = lasso.predict(X_train_scaled)
y_pred_test = lasso.predict(X_test_scaled)

# Visualize predictions vs actual values
plt.scatter(X_test.values, y_test, label='Actual Values')
plt.scatter(X_test.values, y_pred_test, color='red', label='Predicted Values')
plt.xlabel("Median Income")
plt.ylabel("Median House Value")
plt.title("Lasso Regression: Actual vs Predicted Values")
plt.legend()
plt.show()
