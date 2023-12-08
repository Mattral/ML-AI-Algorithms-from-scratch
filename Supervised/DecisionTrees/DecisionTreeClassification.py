import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.tree import plot_tree


class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        """
        Decision tree classifier.

        Parameters:
        - max_depth: int or None, optional (default=None)
            The maximum depth of the tree.
        """
        self.max_depth = max_depth
        self.tree = None
        self.num_classes = None

    def fit(self, X, y):
        """
        Build the decision tree.

        Parameters:
        - X: numpy array, shape (n_samples, n_features)
            Input features.
        - y: numpy array, shape (n_samples,)
            Labels.
        """
        self.num_classes = len(np.unique(y))
        self.tree = self._grow_tree(X, y)

    def _gini(self, y):
        """
        Calculate the Gini index for a set of labels.

        Parameters:
        - y: numpy array
            Array of labels.

        Returns:
        - float
            Gini index.
        """
        m = len(y)
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in np.unique(y))

    def _best_split(self, X, y):
        """
        Find the best split for a node.

        Parameters:
        - X: numpy array, shape (n_samples, n_features)
            Input features.
        - y: numpy array, shape (n_samples,)
            Labels.

        Returns:
        - idx: int
            Index of the feature to split on.
        - thr: float
            The threshold value for the split.
        """
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
        """
        Recursively grow the decision tree.

        Parameters:
        - X: numpy array, shape (n_samples, n_features)
            Input features.
        - y: numpy array, shape (n_samples,)
            Labels.
        - depth: int
            Current depth of the tree.

        Returns:
        - dict
            Tree node.
        """
        num_samples_per_class = [np.sum(y == i) for i in np.unique(y)]
        predicted_class = np.argmax(num_samples_per_class)
        node = {'class': predicted_class, 'num_samples': len(y)}

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
        Predict labels for input features.

        Parameters:
        - X: numpy array, shape (n_samples, n_features)
            Input features.

        Returns:
        - list
            Predicted labels.
        """
        return [self._predict(inputs) for inputs in X]

    def _predict(self, inputs):
        """
        Recursively traverse the tree to predict the label for a single input.

        Parameters:
        - inputs: numpy array
            Input features.

        Returns:
        - int
            Predicted label.
        """
        node = self.tree
        while 'index' in node:
            if inputs[node['index']] < node['threshold']:
                node = node['left']
            else:
                node = node['right']
        return node['class']

    def plot_tree(self, feature_names=None, class_names=None, filled=True):
        """
        Plot the decision tree.

        Parameters:
        - feature_names: list or None, optional (default=None)
            List of feature names.
        - class_names: list or None, optional (default=None)
            List of class names.
        - filled: bool, optional (default=True)
            Whether to fill the boxes with colors.
        """
        print("")
        print("Tree")
        print("Feature Names:", feature_names)
        print("Class Names:", class_names)
        plt.figure(figsize=(10, 10))
        plt.xlabel(feature_names)
        plt.ylabel(class_names)
        self._plot_tree_rec(self.tree, feature_names, class_names, filled)
        plt.show()


    def _plot_tree_rec(self, node, feature_names, class_names, filled=True, indent=0):
        if node is None:
            return

        if 'left' not in node and 'right' not in node:
            label = f"{class_names[node['class']]} (class {node['class']})"
            print(f"{' ' * indent}Leaf: {label}")
            return

        print(f"{' ' * indent}Feature {feature_names[node['index']]} <= {node['threshold']:.2f}")

        print(f"{' ' * indent}Left:")
        self._plot_tree_rec(node['left'], feature_names, class_names, filled, indent + 2)

        print(f"{' ' * indent}Right:")
        self._plot_tree_rec(node['right'], feature_names, class_names, filled, indent + 2)



# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the Iris dataset
print("Iris Dataset:")
print("Features (X):")
print(X_train[:5])  # Print the first 5 samples
print("Labels (y):")
print(y_train[:5])  # Print the first 5 labels

# Create and train the decision tree classifier from scikit-learn
dt_classifier = DecisionTreeClassifier(max_depth=3)
dt_classifier.fit(X_train, y_train)



# Print the decision tree
print("\nDecision Tree:")
print(dt_classifier.tree)

# Predict labels for the test set
y_pred = dt_classifier.predict(X_test)

# Print the predicted labels and true labels
print("\nPredicted Labels:")
print(y_pred)
print("True Labels:")
print(y_test)

# Create and fit the decision tree
dt_classifier = DecisionTreeClassifier(max_depth=3)
dt_classifier.fit(X, y)

#print Tree
dt_classifier._plot_tree_rec(dt_classifier.tree, iris.feature_names, iris.target_names)


# Plot the decision tree

dt_classifier.plot_tree(feature_names=iris.feature_names, class_names=iris.target_names)
