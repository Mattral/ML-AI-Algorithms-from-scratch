import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class LassoRegression:
    def __init__(self, alpha=1.0, max_iterations=1000, tol=1e-4):
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.tol = tol
        self.theta = None

    def soft_threshold(self, x, threshold):
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

    def fit(self, X, y):
        # Add a bias term to X
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]

        # Initialize coefficients
        self.theta = np.random.randn(X_bias.shape[1])

        for _ in range(self.max_iterations):
            old_theta = self.theta.copy()

            # Coordinate Descent for LASSO Regression
            for j in range(X_bias.shape[1]):
                X_j = X_bias[:, j]
                y_pred = X_bias @ self.theta - X_j * self.theta[j]
                rho = X_j.T @ (y - y_pred)
                self.theta[j] = self.soft_threshold(rho, self.alpha)

            # Check for convergence
            if np.linalg.norm(self.theta - old_theta) < self.tol:
                break

    def predict(self, X):
        # Add a bias term to X
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]

        # Make predictions
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
