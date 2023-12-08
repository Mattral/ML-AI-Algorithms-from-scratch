# Decision Tree Classifier and Regressor



## Overview

This repository provides Python implementations of Decision Tree models for classification and regression tasks. Decision Trees are versatile and interpretable machine learning algorithms that recursively partition the input space based on feature values.

## Background

Decision Trees are non-linear models that make decisions based on the values of input features. They consist of nodes representing decision points, branches representing possible outcomes, and leaves representing the final decision or prediction. These structures make Decision Trees intuitive and easy to interpret.

### Classification
Decision Trees for classification assign labels to input samples by traversing the tree from the root to a leaf node based on feature conditions. The majority class in a leaf node determines the predicted class for a given input.

### Regression
Decision Trees for regression predict continuous values by averaging target values in leaf nodes. Similar to classification, the traversal is based on feature conditions, and the leaf node's mean target value is the final prediction.

## How Decision Trees Work

1. **Splitting Criteria:**
   - The algorithm selects the best feature and threshold to split the data based on criteria like Gini impurity for classification or mean squared error for regression.

2. **Recursive Splitting:**
   - The dataset is recursively split into subsets until a stopping criterion is met, such as reaching a maximum depth or a minimum number of samples in a node.

3. **Leaf Node Prediction:**
   - In classification, the majority class in a leaf node is the predicted class. In regression, it's the mean of target values in the leaf.

4. **Tree Visualization:**
   - The trained Decision Tree can be visualized to understand its structure and decision-making process.




## Programs

### Decision Tree Classifier

- **File:** `DecisionTreeClassification.py`
- **Description:**
  - Implements a Decision Tree Classifier from scratch without using external libraries.
  - Capable of training on a dataset and making predictions.
  - Includes a method to visualize the trained decision tree.

- **Usage:**
  ```bash
  python DecisionTreeClassification.py
  ```

  ### Decision Tree Regressor

- **File**: `DecisionTreeRegression.py`
- **Description:**
  - Implements a Decision Tree Regressor from scratch.
  - Capable of training on a dataset and making regression predictions.
  - Includes a method to visualize the trained decision tree.

- **Usage:**
  ```bash
  python DecisionTreeRegression.py
  ```

## Datasets

### Classification Dataset
Name: Iris Dataset
Description: A well-known dataset for classification tasks, included in scikit-learn.

### Regression Dataset
Name: California Housing Dataset
Description: A dataset containing median house values for California districts, 

  

