# Linear Regression Algorithm

## Overview

This repository contains a simple implementation of the linear regression algorithm from scratch using Python and NumPy. Linear regression is a fundamental machine learning algorithm used for predicting a continuous target variable based on one or more input features.


## Background

### What is Linear Regression?

Linear Regression is a supervised machine learning algorithm used for predicting the value of a continuous target variable based on one or more input features. The algorithm models the relationship between the dependent variable (output) and independent variables (features) by fitting a linear equation to the observed data.

### How does Linear Regression work?

The linear equation is represented as:

$$\[ y = \theta_0 + \theta_1 \cdot x_1 + \theta_2 \cdot x_2 + \ldots + \theta_n \cdot x_n \]$$

where:
- \( y \) is the predicted output,
- \( \theta_0 \) is the bias term,
- \( \theta_1, \theta_2, \ldots, \theta_n \) are the weights,
- \( x_1, x_2, \ldots, x_n \) are the input features.

The objective is to find the values of \( \theta \) that minimize the mean squared error (MSE) between the predicted and actual values.


## Implementation Details

### LinearRegression Class

The core of the implementation is the `LinearRegression` class, which encapsulates the linear regression model. Here's a brief overview of its key methods:

#### `__init__(self, learning_rate=0.01, num_iterations=1000)`

- Initializes the linear regression model with default or user-defined learning rate and the number of iterations for gradient descent.

#### `fit(self, X, y)`

- Fits the model to the training data using gradient descent.
- `X`: Input features.
- `y`: Target variable.

#### `predict(self, X)`

- Predicts the target variable for new input features.
- `X`: Input features.

### Workflow

1. **Initialization:** Create an instance of the `LinearRegression` class, specifying hyperparameters if needed.

2. **Data Preparation:** Prepare your training data, ensuring it's in the appropriate format (NumPy arrays).

3. **Model Training:** Call the `fit` method to train the model on the training data.

4. **Prediction:** Use the `predict` method to make predictions on new data.


The optimization objective (mean squared error) is given by the formula:

$\[ J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 \]$

where:
- $[\( J(\theta) \)]$ is the cost function.
- $\( m \)$ is the number of training examples.
- $\( h_\theta(x) \)$ is the hypothesis function.
- $\( x^{(i)} \)$ are the input features for the $\( i \)$-th example.
- $\( y^{(i)} \)$ is the target variable for the \( i \)$-th example.


## Usage

### Synthetic Data Example

```python
import numpy as np
import matplotlib.pyplot as plt
from LinearRegressionAlgo import LinearRegression

# Generate synthetic data for demonstration
np.random.seed(42)
X_synthetic = 2 * np.random.rand(100, 1)
y_synthetic = 4 + 3 * X_synthetic + np.random.randn(100, 1)

# Create and train the linear regression model
model = LinearRegression(learning_rate=0.01, num_iterations=1000)
model.fit(X_synthetic, y_synthetic)

# Make predictions
predictions_synthetic = model.predict(X_synthetic)

# Plot the synthetic data and the linear regression line
plt.scatter(X_synthetic, y_synthetic, label='Synthetic Data')
plt.plot(X_synthetic, predictions_synthetic, color='red', label='Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with Synthetic Data')
plt.legend()
plt.show()
```

The linear regression model predicts the target variable \( y \) using the formula:

$\[ y = \theta_0 + \theta_1 \cdot x \]$

### CSV Data Example

```python
import pandas as pd
from LinearRegressionAlgo import LinearRegression

# Read data from a CSV file
csv_file_path = 'your_dataset.csv'
data = pd.read_csv(csv_file_path)
X_csv = data[['feature']].values
y_csv = data['target'].values

# Create and train the linear regression model using CSV data
model_csv = LinearRegression(learning_rate=0.01, num_iterations=1000)
model_csv.fit(X_csv, y_csv)

# Make predictions using CSV data
predictions_csv = model_csv.predict(X_csv)

```

