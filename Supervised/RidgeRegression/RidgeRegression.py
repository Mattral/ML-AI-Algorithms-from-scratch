
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.theta = None

    def fit(self, X, y):
        # Add a bias term to X
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]

        # Compute the Ridge Regression coefficients using the closed-form solution
        n_features = X_bias.shape[1]
        identity_matrix = np.eye(n_features)
        self.theta = np.linalg.inv(X_bias.T @ X_bias + self.alpha * identity_matrix) @ (X_bias.T @ y)

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

# Create and fit the Ridge Regression model
ridge = RidgeRegression(alpha=1.0)
ridge.fit(X_train_scaled, y_train)

# Predictions
y_pred_train = ridge.predict(X_train_scaled)
y_pred_test = ridge.predict(X_test_scaled)

# Visualize predictions vs actual values
plt.scatter(X_test.values, y_test, label='Actual Values')
plt.scatter(X_test.values, y_pred_test, color='red', label='Predicted Values')
plt.xlabel("Median Income")
plt.ylabel("Median House Value")
plt.title("Ridge Regression: Actual vs Predicted Values")
plt.legend()
plt.show()
