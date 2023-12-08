# K-Nearest Neighbors (KNN) Algorithm

## Overview

K-Nearest Neighbors (KNN) is a simple and intuitive machine learning algorithm used for both classification and regression tasks. It belongs to the family of instance-based, lazy learning algorithms, as it makes predictions based on the majority class or average of the k-nearest data points in the feature space.

## Implementation

The provided Python script (`KNNClassifier.py`) demonstrates the implementation of KNN from scratch without relying on external libraries such as scikit-learn. It includes examples of both KNN classification and regression using the housing dataset.

### KNN Classification

In the classification example, the algorithm predicts whether the `median_house_value` is above or below the median value based on the `median_income`. It achieves this by finding the k-nearest neighbors in the feature space and assigning the most common class label among them.

### KNN Regression

In the regression example, the algorithm predicts the `median_house_value` by averaging the target values of the k-nearest neighbors in the feature space.

## How It Works

1. **Distance Calculation:**
   - The algorithm calculates the distance between the input data point and all data points in the training set. Common distance metrics include Euclidean distance, Manhattan distance, or Minkowski distance.

2. **Neighbor Selection:**
   - It identifies the k-nearest neighbors based on the calculated distances.

3. **Majority Voting (Classification) or Averaging (Regression):**
   - For classification, the algorithm assigns the most common class label among the k-nearest neighbors.
   - For regression, it predicts the average value of the target variable for the k-nearest neighbors.

## Advantages

- **Simplicity:** KNN is easy to understand and implement, making it a good choice for quick prototyping.
- **No Training Phase:** KNN is a lazy learner, meaning it doesn't have a separate training phase. The model is built during prediction time.

## Disadvantages

- **Computational Cost:** As the dataset grows, the computational cost of finding the nearest neighbors increases.
- **Sensitive to Outliers:** Outliers can significantly affect the performance of KNN.
- **Curse of Dimensionality:** KNN is less effective in high-dimensional spaces due to the increased sparsity of data.

## Common Use Cases

- **Classification:** KNN is often used in image recognition, document classification, and recommendation systems.
- **Regression:** It can be applied to predict numerical values, such as house prices.
- **Anomaly Detection:** KNN can be used to identify outliers or anomalies in data.

## Usage

1. Run the provided Python script:
   ```bash
   python KNNClassifier.py
