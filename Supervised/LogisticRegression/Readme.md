# Logistic Regression

## Overview

This repository contains a Python implementation of the Logistic Regression algorithm from scratch. Logistic Regression is a popular machine learning algorithm used for binary classification problems.


## Background

### What is Logistic Regression?

Logistic Regression is a statistical method used for predicting the probability of a binary outcome. It is particularly useful for classification tasks where the target variable is categorical and has two classes. The algorithm models the relationship between the dependent variable (output) and one or more independent variables (features) by estimating probabilities using the logistic function.

### How does Logistic Regression work?

The logistic function, also known as the sigmoid function, is used to map any real-valued number into a value between 0 and 1. The predicted probability is then converted into a binary outcome based on a threshold.

The key equations are:

$$\[ z = \theta_0 + \theta_1 \cdot x_1 + \theta_2 \cdot x_2 + \ldots + \theta_n \cdot x_n \]$$

$$\[ h(z) = \frac{1}{1 + e^{-z}} \]$$

where:
- $\( z \)$ is the linear combination of weights (\( \theta \)) and features (\( x \)),
- $\( h(z) \)$ is the sigmoid function.

The weights and bias are updated iteratively using gradient descent to minimize the binary cross-entropy loss.

## Implementation

The Logistic Regression algorithm is implemented in Python using NumPy for numerical operations. The implementation includes methods for training the model, making predictions, and calculating the binary cross-entropy loss.

### File Structure

- `LogisticRegressionAlgo.py`: Main implementation of the Logistic Regression algorithm.
- `unitTestLogReg.py`: Unit tests for the Logistic Regression class.

## Usage

To use the Logistic Regression algorithm, follow these steps:

1. Import the `LogisticRegression` class.
2. Create an instance of the class with desired hyperparameters (learning rate, number of iterations).
3. Train the model using the `fit` method with your training data.
4. Make predictions using the `predict` method.


## Example

```python
import numpy as np
import matplotlib.pyplot as plt
from LogisticRegression import LogisticRegression

# Generate synthetic data for binary classification
np.random.seed(42)
X_synthetic = 2 * np.random.rand(100, 2)
y_synthetic = (X_synthetic[:, 0] + X_synthetic[:, 1] > 2).astype(int)

# Create and train the logistic regression model
model = LogisticRegression(learning_rate=0.01, num_iterations=1000)
model.fit(X_synthetic, y_synthetic)

# Make predictions on the training data
predictions_synthetic = model.predict(X_synthetic)

# Plot the synthetic data and decision boundary
plt.scatter(X_synthetic[:, 0], X_synthetic[:, 1], c=y_synthetic, cmap='viridis', marker='o', edgecolors='k', label='Data Points')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Plot the decision boundary
x_decision_boundary = np.linspace(0, 2, 100)
y_decision_boundary = -(model.weights[0] * x_decision_boundary + model.bias) / model.weights[1]
plt.plot(x_decision_boundary, y_decision_boundary, color='red', label='Decision Boundary')

plt.title('Logistic Regression with Synthetic Data')
plt.legend()
plt.show()

```

###Unit Tests

The unitTestLogReg.py file contains unit tests to ensure the correctness of the Logistic Regression implementation. Run the tests using your preferred testing framework.
