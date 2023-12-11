import numpy as np
from sklearn import datasets
import pandas as pd



def target_distribution(x, y):
    """
    Define a simple joint distribution.

    Parameters:
    - x (float): The x-coordinate.
    - y (float): The y-coordinate.

    Returns:
    - float: The value of the joint distribution at the specified (x, y).
    """
    return np.exp(-(x**2 + y**2))

def gibbs_sampling(iterations):
    """
    Perform Gibbs Sampling to generate samples from a joint distribution.

    Parameters:
    - iterations (int): The number of iterations for sampling.

    Returns:
    - np.ndarray: An array containing generated samples, where each row corresponds to (x, y).
    """
    # Initial values
    x = 0.0
    y = 0.0

    samples = []

    for _ in range(iterations):
        # Update x based on the current value of y
        x = np.random.normal(loc=y, scale=1.0)

        # Update y based on the new value of x
        y = np.random.normal(loc=x, scale=1.0)

        samples.append((x, y))

    return np.array(samples)

if __name__ == "__main__":
    # Number of iterations
    iterations = 1000

    # Run Gibbs Sampling
    samples = gibbs_sampling(iterations)

    # Display results
    print("Gibbs Sampling Results:")
    print(f"Mean of x: {np.mean(samples[:, 0]):.4f}")
    print(f"Mean of y: {np.mean(samples[:, 1]):.4f}")



'''
----------------------- IRIS DATASET ------------------
'''

# Load the Iris dataset
iris = datasets.load_iris()

# Convert the dataset to a Pandas DataFrame for easy exploration
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target_names[iris.target]

# Display the first few rows of the dataset
print(iris_df.head())

# Display information about the dataset
print("\nDataset Information:")
print(iris_df.info())

# Display statistical summary of the dataset
print("\nStatistical Summary:")
print(iris_df.describe())

# Display the distribution of classes
print("\nClass Distribution:")
print(iris_df['target'].value_counts())
