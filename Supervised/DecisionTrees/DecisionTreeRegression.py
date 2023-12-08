import numpy as np
import pandas as pd

# Load the dataset
data = pd.read_csv("housing.csv")
X = data["median_income"].values.reshape(-1, 1)  # Feature: median_income
y = data["median_house_value"].values  # Target: median_house_value

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree Regressor class
class DecisionTreeRegressor:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y, depth=0)

    def _mse(self, y):
        if len(y) == 0:
            return 0
        mean = np.mean(y)
        return np.mean((y - mean) ** 2)

    def _best_split(self, X, y):
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
        node = {'value': np.mean(y), 'num_samples': len(y)}

        if self.max_depth is None or depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node['index'] = idx
                node['threshold'] = thr

                print(f"Depth: {depth}, Splitting at X[{idx}] <= {thr:.2f}, MSE: {self._mse(y):.4f}")

                print(f"Depth: {depth + 1}, Left Child - Samples: {len(y_left)}, Mean Value: {np.mean(y_left):.4f}")
                node['left'] = self._grow_tree(X_left, y_left, depth + 1)

                print(f"Depth: {depth + 1}, Right Child - Samples: {len(y_right)}, Mean Value: {np.mean(y_right):.4f}")
                node['right'] = self._grow_tree(X_right, y_right, depth + 1)

        return node

    def predict(self, X):
        return [self._predict(inputs) for inputs in X]

    def _predict(self, inputs):
        node = self.tree
        while 'index' in node:
            if inputs[node['index']] < node['threshold']:
                node = node['left']
            else:
                node = node['right']
        return node['value']

# Create and train the decision tree regressor
dt_regressor = DecisionTreeRegressor(max_depth=None)
dt_regressor.fit(X_train, y_train)

# Predict target values for the test set
y_pred = dt_regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
