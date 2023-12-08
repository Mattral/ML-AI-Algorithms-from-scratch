# Running this program will take some time on each iteration

'''
MSE is calculated as the average of the squared differences between predicted and actual values.
The fact that this value is large doesn't necessarily indicate a problem.
It simply means that, on average, the squared differences between your predicted and actual values are large.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class DecisionTreeRegressor:
    def __init__(self, max_depth=None):
        """
        Decision tree regressor.

        Parameters:
        - max_depth: int or None, optional (default=None)
            The maximum depth of the tree.
        """
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        """
        Build the decision tree.

        Parameters:
        - X: numpy array, shape (n_samples, n_features)
            Input features.
        - y: numpy array, shape (n_samples,)
            Target values.
        """
        self.tree = self._grow_tree(X, y, depth=0)

    def _mse(self, y):
        """
        Calculate the mean squared error for a set of values.

        Parameters:
        - y: numpy array
            Array of target values.

        Returns:
        - float
            Mean squared error.
        """
        if len(y) == 0:
            return 0
        mean = np.mean(y)
        return np.mean((y - mean) ** 2)

    def _best_split(self, X, y):
        """
        Find the best split for a node.

        Parameters:
        - X: numpy array, shape (n_samples, n_features)
            Input features.
        - y: numpy array, shape (n_samples,)
            Target values.

        Returns:
        - idx: int
            Index of the feature to split on.
        - thr: float
            The threshold value for the split.
        """
        m = len(y)
        if m <= 1:
            return None, None

        mse_parent = self._mse(y)

        best_mse = mse_parent
        best_idx, best_thr = None, None

        for idx in range(X.shape[1]):
            thresholds, values = zip(*sorted(zip(X[:, idx], y)))

            num_left = 0
            num_right = m

            sum_left = 0
            sum_right = np.sum(y)

            for i in range(1, m):
                num_left += 1
                num_right -= 1

                sum_left += values[i - 1]
                sum_right -= values[i - 1]

                mse_left = self._mse(values[:i])
                mse_right = self._mse(values[i:])

                mse = (num_left * mse_left + num_right * mse_right) / m

                if mse < best_mse:
                    best_mse = mse
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2

        return best_idx, best_thr

    def _grow_tree(self, X, y, depth=0):
        """
        Recursively grow the decision tree.

        Parameters:
        - X: numpy array, shape (n_samples, n_features)
            Input features.
        - y: numpy array, shape (n_samples,)
            Target values.
        - depth: int
            Current depth of the tree.

        Returns:
        - dict
            Tree node.
        """
        node = {'value': np.mean(y), 'num_samples': len(y)}

        if self.max_depth is None or depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node['index'] = idx
                node['threshold'] = thr
                node['left'] = self._grow_tree(X_left, y_left, depth + 1)
                node['right'] = self._grow_tree(X_right, y_right, depth + 1)

        return node

    def predict(self, X):
        """
        Predict target values for input features.

        Parameters:
        - X: numpy array, shape (n_samples, n_features)
            Input features.

        Returns:
        - numpy array
            Predicted target values.
        """
        return [self._predict(inputs) for inputs in X]

    def _predict(self, inputs):
        """
        Recursively traverse the tree to predict the target value for a single input.

        Parameters:
        - inputs: numpy array
            Input features.

        Returns:
        - float
            Predicted target value.
        """
        node = self.tree
        while 'index' in node:
            if inputs[node['index']] < node['threshold']:
                node = node['left']
            else:
                node = node['right']
        return node['value']

class RandomForestRegressor:
    def __init__(self, n_trees=100, max_depth=None):
        """
        Random Forest regressor.

        Parameters:
        - n_trees: int, optional (default=100)
            The number of trees in the forest.
        - max_depth: int or None, optional (default=None)
            The maximum depth of each tree in the forest.
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        """
        Build the Random Forest.

        Parameters:
        - X: numpy array, shape (n_samples, n_features)
            Input features.
        - y: numpy array, shape (n_samples,)
            Target values.
        """
        for _ in range(self.n_trees):
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            indices = np.random.choice(len(X), len(X), replace=True)
            tree.fit(X[indices], y[indices])
            self.trees.append(tree)

    def predict(self, X):
        """
        Predict target values for input features.

        Parameters:
        - X: numpy array, shape (n_samples, n_features)
            Input features.

        Returns:
        - numpy array
            Predicted target values.
        """
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(predictions, axis=0)

# Load the housing dataset
housing_data = pd.read_csv("housing.csv")
X_housing = housing_data["median_income"].values.reshape(-1, 1)
y_housing = housing_data["median_house_value"].values

# Split the data into training and testing sets
X_train_housing, X_test_housing, y_train_housing, y_test_housing = train_test_split(X_housing, y_housing, test_size=0.2, random_state=42)

# Create and train the Random Forest Regressor from scratch with prints
rf_regressor_scratch_with_prints = RandomForestRegressor(n_trees=100, max_depth=3)

for i in range(rf_regressor_scratch_with_prints.n_trees):
    tree = DecisionTreeRegressor(max_depth=rf_regressor_scratch_with_prints.max_depth)
    indices = np.random.choice(len(X_train_housing), len(X_train_housing), replace=True)
    tree.fit(X_train_housing[indices], y_train_housing[indices])
    rf_regressor_scratch_with_prints.trees.append(tree)

    # Predict target values for the test set
    y_pred_scratch_with_prints = rf_regressor_scratch_with_prints.predict(X_test_housing)

    # Print information about the current iteration
    print(f"\nIteration {i + 1}:\n")

    # Evaluate the model
    mse_scratch_with_prints = mean_squared_error(y_test_housing, y_pred_scratch_with_prints)
    print(f"Mean Squared Error (Random Forest from Scratch): {mse_scratch_with_prints}")

    # Plot the predicted values against the true values
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test_housing, y_test_housing, color='blue', label='True Values')
    plt.scatter(X_test_housing, y_pred_scratch_with_prints, color='red', label='Predicted Values')
    plt.title(f"Random Forest Regression - Iteration {i + 1}")
    plt.xlabel("Median Income")
    plt.ylabel("Median House Value")
    plt.legend()
    plt.show()
