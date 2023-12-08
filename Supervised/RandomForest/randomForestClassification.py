"""
Decision Tree and Random Forest Classifier Implementation

This script implements a basic Decision Tree and Random Forest Classifier from scratch
using NumPy and pandas. It also utilizes the scikit-learn library for some functionalities.

The DecisionTreeClassifier class implements a basic decision tree for classification tasks.
The RandomForestClassifier class builds an ensemble of decision trees to improve predictive performance.

Note: This implementation is for educational purposes. For real-world applications, it is recommended
to use the scikit-learn library, which is optimized and well-tested.

Requirements:
- NumPy
- pandas
- scikit-learn
- matplotlib

Usage:
1. Load the Titanic dataset (CSV file assumed to be named 'titanic.csv').
2. Preprocess the data, handling missing values, and encoding categorical variables.
3. Extract features and labels.
4. Create an instance of RandomForestClassifier and fit it to the training data.
5. Evaluate the model on the test set.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree  
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier


class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        """
        Decision Tree Classifier Implementation.

        Parameters:
        - max_depth: int or None, optional (default=None)
            The maximum depth of the tree. If None, nodes are expanded until
            they contain less than the minimum samples required to split.

        Attributes:
        - max_depth: int or None
            The specified maximum depth of the tree.
        - tree: sklearn.tree.DecisionTreeClassifier
            The underlying scikit-learn DecisionTreeClassifier model.
        - num_classes: int
            The number of unique classes in the target variable.
        - feature_importances_: numpy array
            Feature importances computed from the underlying Decision Tree model.
        """
        self.max_depth = max_depth
        self.tree = None
        self.num_classes = None
        self.feature_importances_ = np.zeros(1)

        self.feature_importances_ = np.zeros(1)

    def fit(self, X, y):
        """
        Fit the Decision Tree Classifier to the training data.

        Parameters:
        - X: numpy array, shape (n_samples, n_features)
            The training input samples.
        - y: numpy array, shape (n_samples,)
            The target values.

        Returns:
        - None
        """
        self.num_classes = len(np.unique(y))
        self.tree = SklearnDecisionTreeClassifier(max_depth=self.max_depth)
        self.tree.fit(X, y)
        self.feature_importances_ = self.tree.feature_importances_

    def _gini(self, y):
        m = len(y)
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in np.unique(y))

    def _best_split(self, X, y):
        m = len(y)
        if m <= 1:
            return None, None

        num_parent = [np.sum(y == c) for c in range(self.num_classes)]

        best_gini = 1.0 - sum((num / m) ** 2 for num in num_parent)
        best_idx, best_thr = None, None

        for idx in range(X.shape[1]):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))

            num_left = [0] * self.num_classes
            num_right = num_parent.copy()

            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(self.num_classes))
                gini_right = 1.0 - sum((num_right[x] / (m - i)) ** 2 for x in range(self.num_classes))

                gini = (i * gini_left + (m - i) * gini_right) / m

                if thresholds[i] == thresholds[i - 1]:
                    continue

                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2

        return best_idx, best_thr

    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in np.unique(y)]
        predicted_class = np.argmax(num_samples_per_class)
        node = {'class': predicted_class, 'num_samples': len(y)}

        feature_importances = np.zeros(X.shape[1])

        if self.max_depth is None or depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node['index'] = idx
                node['threshold'] = thr
                left_node, left_importance = self._grow_tree(X_left, y_left, depth + 1)
                right_node, right_importance = self._grow_tree(X_right, y_right, depth + 1)
                node['left'] = left_node
                node['right'] = right_node

                # Make sure left_importance and right_importance are single values
                if not np.isscalar(left_importance):
                    left_importance = np.sum(left_importance)
                if not np.isscalar(right_importance):
                    right_importance = np.sum(right_importance)

                # Calculate feature importances
                feature_importances[idx] += left_importance
                feature_importances[idx] += right_importance

        return node, feature_importances

    def predict(self, X):
        return self.tree.predict(X)

    def _predict(self, inputs):
        node = self.tree
        while 'index' in node:
            if inputs[node['index']] < node['threshold']:
                node = node['left']
            else:
                node = node['right']
        return node['class']

class RandomForestClassifier:
    def __init__(self, n_trees=100, max_depth=None):
        """
        Random Forest Classifier Implementation.

        Parameters:
        - n_trees: int, optional (default=100)
            The number of trees in the forest.
        - max_depth: int or None, optional (default=None)
            The maximum depth of the trees. If None, nodes are expanded until
            they contain less than the minimum samples required to split.

        Attributes:
        - n_trees: int
            The specified number of trees in the forest.
        - max_depth: int or None
            The specified maximum depth of the trees.
        - trees: list of DecisionTreeClassifier
            The list of DecisionTreeClassifier models forming the Random Forest.
        - feature_importances_: list of numpy arrays
            Feature importances computed from each Decision Tree in the forest.
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []
        self.feature_importances_ = []

    def fit(self, X, y):

        """
        Fit the Random Forest Classifier to the training data.

        Parameters:
        - X: numpy array, shape (n_samples, n_features)
            The training input samples.
        - y: numpy array, shape (n_samples,)
            The target values.

        Returns:
        - None
        """
        for i in range(self.n_trees):
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            indices = np.random.choice(len(X), len(X), replace=True)
            tree.fit(X[indices], y[indices])
            self.trees.append(tree)

            # Print information about the current iteration
            print(f"Iteration {i + 1} - Tree Depth: {tree.max_depth}")

            # Append the feature importance to the list
            if hasattr(tree, 'feature_importances_'):
                if not hasattr(self, 'feature_importances_'):
                    self.feature_importances_ = []
                self.feature_importances_.append(tree.feature_importances_[0])

            # Visualize the first tree in each iteration
            if i == 0:
                plt.figure(figsize=(15, 8))
                plot_tree(tree.tree, filled=True, feature_names=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'], class_names=['Not Survived', 'Survived'])
                plt.title(f'Decision Tree - Iteration {i + 1}')
                plt.show()

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(predictions, axis=0).astype(int)





# Load the Titanic dataset
titanic_data = pd.read_csv("titanic.csv")

# Preprocess the data (you may need to handle missing values and encode categorical variables)

# Extract features and labels
X = titanic_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = titanic_data['Survived']

# Convert categorical variables to numerical
X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})

# Handle missing values (you may need to impute or drop missing values)
X.fillna(0, inplace=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)

# Create and train the Random Forest Classifier from scratch
rf_classifier = RandomForestClassifier(n_trees=5, max_depth=None)  # Reduced to 5 trees for quicker visualization
rf_classifier.fit(X_train, y_train)

# Predict labels for the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Final Accuracy: {accuracy}")
# Predict labels for the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Final Accuracy: {accuracy}")
