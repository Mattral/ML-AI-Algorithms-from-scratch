import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def generate_synthetic_data(size=100):
    """
    Generate synthetic data for testing Bayesian Linear Regression.

    Parameters:
    - size (int): Number of data points.

    Returns:
    - X (ndarray): Input feature vector.
    - y (ndarray): Output labels.
    """
    np.random.seed(42)
    X = np.linspace(0, 10, size)[:, np.newaxis]
    y = 2 * X.flatten() + 1 + np.random.normal(0, 2, size)
    return X, y

def bayesian_linear_regression(X, y, alpha, beta):
    """
    Bayesian Linear Regression implementation.

    Parameters:
    - X (ndarray): Input feature vector.
    - y (ndarray): Output labels.
    - alpha (float): Prior precision.
    - beta (float): Noise precision.

    Returns:
    - mean (ndarray): Mean of the predictive distribution.
    - cov (ndarray): Covariance matrix of the predictive distribution.
    """
    # Design matrix
    Phi = np.c_[np.ones_like(X), X]

    # Prior precision matrix
    S0_inv = alpha * np.eye(2)

    # Noise precision
    SN_inv = beta * np.eye(len(X))

    # Posterior precision matrix
    SN = np.linalg.inv(np.linalg.inv(S0_inv) + Phi.T @ SN_inv @ Phi)

    # Posterior mean
    mN = beta * SN @ Phi.T @ SN_inv @ y

    return mN, SN

def predict(X_pred, mN, SN, beta):
    """
    Predict using Bayesian Linear Regression.

    Parameters:
    - X_pred (ndarray): Input feature vector for prediction.
    - mN (ndarray): Posterior mean.
    - SN (ndarray): Posterior covariance matrix.
    - beta (float): Noise precision.

    Returns:
    - mean (ndarray): Mean of the predictive distribution.
    - std (ndarray): Standard deviation of the predictive distribution.
    """
    # Design matrix for prediction
    Phi_pred = np.c_[np.ones_like(X_pred), X_pred]

    # Predictive mean
    mean = Phi_pred @ mN

    # Predictive variance
    var = 1 / beta + np.sum(Phi_pred @ SN * Phi_pred, axis=1)
    std = np.sqrt(var)

    return mean, std

def plot_regression_line(X, y, mean, std):
    """
    Plot Bayesian Linear Regression results.

    Parameters:
    - X (ndarray): Input feature vector.
    - y (ndarray): Output labels.
    - mean (ndarray): Mean of the predictive distribution.
    - std (ndarray): Standard deviation of the predictive distribution.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', s=5, label='Data Points')
    plt.plot(X, mean, color='red', label='Mean Prediction')
    plt.fill_between(X.flatten(), mean - 2 * std, mean + 2 * std, color='orange', alpha=0.3, label='Uncertainty')
    plt.title('Bayesian Linear Regression')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()

# Generate synthetic data
X, y = generate_synthetic_data()

# Set hyperparameters
alpha = 1.0  # Prior precision
beta = 0.1   # Noise precision

# Perform Bayesian Linear Regression
mN, SN = bayesian_linear_regression(X, y, alpha, beta)

# Generate new data for prediction
X_pred = np.linspace(0, 10, 100)[:, np.newaxis]

# Make predictions
mean, std = predict(X_pred, mN, SN, beta)

# Visualize results
plot_regression_line(X, y, mean, std)




