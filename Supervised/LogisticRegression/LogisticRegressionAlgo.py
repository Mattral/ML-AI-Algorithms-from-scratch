import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        """
        Initialize the logistic regression model.

        Parameters:
        - learning_rate (float): The learning rate for gradient descent.
        - num_iterations (int): The number of iterations for gradient descent.
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def sigmoid(self, z):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-z))

    def calculate_loss(self, y, predictions):
        """Calculate the binary cross-entropy loss."""
        return -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))

    def fit(self, X, y):
        """
        Train the logistic regression model.

        Parameters:
        - X (numpy array): Input features.
        - y (numpy array): Target labels (0 or 1).
        """
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)  # Ensure y is a column vector
        m, n = X.shape

        self.weights = np.zeros((n, 1))
        self.bias = 0

        for iteration in range(self.num_iterations):
            # Compute the predicted probabilities
            z = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(z)

            # Compute the gradient and update parameters using gradient descent
            errors = predictions - y
            gradient_weights = np.dot(X.T, errors) / m
            gradient_bias = np.sum(errors) / m

            self.weights -= self.learning_rate * gradient_weights
            self.bias -= self.learning_rate * gradient_bias

            # Calculate and store the loss for monitoring
            loss = self.calculate_loss(y, predictions)
            self.loss_history.append(loss)

            if iteration % 100 == 0:
                print(f'Iteration {iteration}, Loss: {loss}')

    def predict(self, X):
        """
        Make predictions using the trained model.

        Parameters:
        - X (numpy array): Input features.

        Returns:
        - numpy array: Predicted labels (0 or 1).
        """
        X = np.array(X)
        z = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(z)
        return np.round(predictions).astype(int)

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
