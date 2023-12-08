import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class PCA:
    def __init__(self, n_components):
        """
        Initialize the PCA object.

        Parameters:
        - n_components (int): Number of principal components to retain.
        """
        self.n_components = n_components
        self.mean = None
        self.projection_matrix = None

    def fit_transform(self, X):
        """
        Fit the PCA model and transform the input data.

        Parameters:
        - X (numpy.ndarray): Input data.

        Returns:
        - X_pca (numpy.ndarray): Transformed data in the reduced space.
        """
        # Ensure X is a 2D array
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Standardize the data
        X_std = StandardScaler().fit_transform(X)

        # Calculate the covariance matrix
        cov_matrix = np.cov(X_std, rowvar=False)

        # Perform eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort eigenvalues and corresponding eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Choose the top k principal components
        self.projection_matrix = eigenvectors[:, :self.n_components]

        # Project the data onto the lower-dimensional subspace
        X_pca = X_std.dot(self.projection_matrix)

        # Save the mean for inverse_transform
        self.mean = np.mean(X, axis=0)

        return X_pca

    def inverse_transform(self, X_pca):
        """
        Inverse transform PCA-reduced data back to the original space.

        Parameters:
        - X_pca (numpy.ndarray): PCA-reduced data.

        Returns:
        - X_original (numpy.ndarray): Inverse-transformed data in the original space.
        """
        # Project back to the original space
        X_original = X_pca.dot(self.projection_matrix.T)

        # Add back the mean
        X_original = X_original + self.mean

        return X_original

# Load the housing dataset from CSV file
df = pd.read_csv('housing.csv')

# Extract features and target variable
X = df[['median_income', 'median_house_value']].values

# Instantiate the PCA class with the desired number of components
pca = PCA(n_components=2)

# Fit and transform the data
X_pca = pca.fit_transform(X)

# Visualize the results
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['median_house_value'], cmap='viridis', alpha=0.5)
plt.title('PCA on Housing Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Median House Value')
plt.show()
