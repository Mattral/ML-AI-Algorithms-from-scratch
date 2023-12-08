'''
This code loads the stock data from the Microsoft_Stock.csv file,
extracts relevant columns as mixed signals, and then applies FastICA
to recover independent components.
The resulting plot shows the original signals and the recovered signals. 
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def center_data(X):
    """Center the data by subtracting the mean."""
    mean = np.mean(X, axis=1, keepdims=True)
    return X - mean

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def fastICA(X, max_iter=1000, tol=1e-4):
    """
    Perform FastICA to recover independent components from mixed signals.

    Parameters:
    - X: numpy array, shape (n_samples, n_features)
        Input data matrix containing mixed signals.
    - max_iter: int, optional (default=1000)
        Maximum number of iterations for the FastICA algorithm.
    - tol: float, optional (default=1e-4)
        Tolerance to declare convergence.

    Returns:
    - W: numpy array, shape (n_features, n_features)
        Demixing matrix.
    - S: numpy array, shape (n_samples, n_features)
        Estimated independent components.
    """

    # Center the data
    X = center_data(X)

    # Initialize random demixing matrix
    W = np.random.rand(X.shape[0], X.shape[0])

    for iteration in range(max_iter):
        # Compute the estimated sources
        S = np.dot(W, X)

        # Compute the contrast function and its gradient
        g = sigmoid(S)
        g_prime = 1 - 2 * g

        # Update the demixing matrix
        W_new = np.dot(g, S.T) / X.shape[1] - np.diag(np.mean(g_prime, axis=1)) @ W

        # Decorrelate the rows of the new demixing matrix
        U, _, Vt = np.linalg.svd(W_new)
        W_new = np.dot(U, Vt)

        # Check for convergence
        if np.max(np.abs(np.abs(np.diag(np.dot(W_new, W.T))) - 1)) < tol:
            break

        W = W_new

    S = np.dot(W, X)
    return W, S

# Load the Microsoft stock data
data = pd.read_csv('Microsoft_Stock.csv')
# Extract relevant columns as mixed signals
X = data[['Open', 'High', 'Low', 'Close', 'Volume']].values.T

# Apply FastICA to recover independent components
W, S = fastICA(X)

# Plot the original signals and recovered signals
plt.figure(figsize=(10, 6))

for i in range(X.shape[0]):
    plt.subplot(X.shape[0], 2, 2*i+1)
    plt.title(f"Original Signal {i+1}")
    plt.plot(X[i, :])

    plt.subplot(X.shape[0], 2, 2*i+2)
    plt.title(f"Recovered Signal {i+1}")
    plt.plot(S[i, :])

plt.tight_layout()
plt.show()
