# Random Forest Implementation

This repository contains Python implementations of Random Forest algorithms for both classification and regression tasks.

## Table of Contents
- [Random Forest Classification](#random-forest-classification)
  - [How it Works](#how-it-works)
  - [Implementation Details](#implementation-details)
  - [Usage](#usage)
- [Random Forest Regression](#random-forest-regression)
  - [How it Works](#how-it-works-1)
  - [Implementation Details](#implementation-details-1)
  - [Usage](#usage-1)

## Random Forest Classification

### How it Works
Random Forest is an ensemble learning algorithm that operates by constructing a multitude of decision trees during training and outputs the class that is the mode of the classes (classification) of the individual trees. The "random" in Random Forest comes from the algorithm's use of random subsamples of the dataset to train each decision tree.

1. **Bootstrapped Sample**: A random subset of the training data is sampled with replacement, creating a bootstrapped sample for each tree.

2. **Random Subset of Features**: At each split in each decision tree, a random subset of features is considered to find the best split.

3. **Decision Trees**: Multiple decision trees are trained independently on different bootstrapped samples and subsets of features.

4. **Voting**: For classification, each tree "votes" on the predicted class, and the mode of the votes is the final predicted class.

### Implementation Details
- The `DecisionTreeClassifier` class implements a basic decision tree for classification.
- The `RandomForestClassifier` class builds an ensemble of decision trees for classification.

### Usage
```python
# Example Usage for Random Forest Classification
from RandomForestClassification import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load your dataset and preprocess it
# ...

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_trees=100, max_depth=None)
rf_classifier.fit(X_train, y_train)

# Predict labels for the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Final Accuracy: {accuracy}")
```



###Random Forest Regression

##How it Works

Random Forest can also be used for regression tasks. In regression, the output is a continuous value instead of a class label.

1. **Bootstrapped Sample**: Similar to classification, a random subset of the training data is sampled with replacement.

2. **Random Subset of Features**: At each split in each decision tree, a random subset of features is considered to find the best split.

3. **Decision Trees**: Multiple decision trees are trained independently on different bootstrapped samples and subsets of features.

4. **Averaging**: For regression, the final prediction is the average of the predictions from all trees.

  ### Implementation Details
- The DecisionTreeRegressor class implements a basic decision tree for regression.
- The RandomForestRegressor class builds an ensemble of decision trees for regression.

###Usage

```python
  # Example Usage for Random Forest Regression
from RandomForestRegression import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load your dataset and preprocess it
# ...

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_trees=100, max_depth=None)
rf_regressor.fit(X_train, y_train)

# Predict values for the test set
y_pred = rf_regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

  ```
