import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class LinearRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, print_interval=None):
        """
        Linear regression model.

        Parameters:
        - learning_rate: float, optional (default=0.01)
            The learning rate determines the step size in updating the model parameters during gradient descent.
            Adjusting it can affect the convergence speed.
            
        - num_iterations: int, optional (default=1000)
            The number of iterations specifies how many times the gradient descent loop will be executed.
            More iterations may lead to a more accurate model, but there's a trade-off with computational cost.
            
        - print_interval: int or None, optional (default=None)
            If provided, print the training progress (loss, parameters) every `print_interval` iterations.
            Set to None to disable printing during training.

        Attributes:
        - learning_rate: float
            The learning rate used for gradient descent.
            
        - num_iterations: int
            The number of iterations for gradient descent.
            
        - weights: numpy array, shape (n_features,)
            Model weights for each feature.
            
        - bias: float
            Model bias term.
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.print_interval = print_interval

    def fit(self, X, y):
        """
        Fit the linear regression model to the training data.

        Parameters:
        - X: numpy array, shape (n_samples, n_features)
            Input features.
            
        - y: numpy array, shape (n_samples,)
            Target variable.

        Returns:
        None
        """
        X = np.array(X)
        y = np.array(y).flatten()
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for iteration in range(self.num_iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            errors = y_pred - y

            # Update weights and bias using gradient descent
            self.weights -= self.learning_rate * (1 / X.shape[0]) * np.dot(X.T, errors)
            self.bias -= self.learning_rate * (1 / X.shape[0]) * np.sum(errors)

            # Print training progress
            if self.print_interval and iteration % self.print_interval == 0:
                loss = self.calculate_loss(X, y)
                print(f"Iteration {iteration}, Loss: {loss}")
                print("Weights:", self.weights)
                print("Bias:", self.bias)

    def predict(self, X):
        """
        Make predictions using the trained linear regression model.

        Parameters:
        - X: numpy array, shape (n_samples, n_features)
            Input features for prediction.

        Returns:
        - y_pred: numpy array, shape (n_samples,)
            Predicted values.
        """
        X = np.array(X)
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred.reshape(-1, 1)

    def calculate_loss(self, X, y):
        """
        Calculate the mean squared error loss for the current model.

        Parameters:
        - X: numpy array, shape (n_samples, n_features)
            Input features.
            
        - y: numpy array, shape (n_samples,)
            Target variable.

        Returns:
        - loss: float
            Mean squared error loss.
        """
        y_pred = np.dot(X, self.weights) + self.bias
        errors = y_pred - y
        loss = np.mean(errors**2)
        return loss

# Read data from a CSV file
# Assuming the CSV file has columns "median_income" and "median_house_value"
csv_file_path = 'housing.csv'  # Replace with the actual path to your CSV file
data = pd.read_csv(csv_file_path)

# Select input features and target variable
X_csv = data[['median_income']].values  # Input features (median income)
y_csv = data['median_house_value'].values  # Target variable (median house value)

# Create and train the linear regression model using CSV data
model_csv = LinearRegression(learning_rate=0.01, num_iterations=1000, print_interval=100)
model_csv.fit(X_csv, y_csv)

# Make predictions using CSV data
predictions_csv = model_csv.predict(X_csv)

# Plot the CSV data and the linear regression line
plt.scatter(X_csv, y_csv, label='CSV Data')  # Scatter plot of the original data
plt.plot(X_csv, predictions_csv, color='green', label='Linear Regression')  # Plot the linear regression line
plt.xlabel('Median Income')
plt.ylabel('Median House Value')
plt.title('Linear Regression with CSV Data')
plt.legend()
plt.show()
