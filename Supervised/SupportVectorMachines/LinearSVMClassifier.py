import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load the housing dataset from CSV file
df = pd.read_csv('housing.csv')

# Extract features and target variable
X = df[['median_income']].values
y = df['median_house_value'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear SVM Classifier from Scratch
class LinearSVMClassifier:
    def __init__(self, learning_rate=0.01, epochs=1000, C=1.0):
        """
        Initialize a Linear SVM Classifier.

        Parameters:
        - learning_rate (float): Learning rate for gradient descent.
        - epochs (int): Number of training epochs.
        - C (float): Regularization parameter.
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.C = C

    def fit(self, X, y):
        """
        Train the Linear SVM Classifier.

        Parameters:
        - X (numpy.ndarray): Training feature data.
        - y (numpy.ndarray): Training target data.
        """
        # Add bias term to X
        X = np.column_stack((np.ones(len(X)), X))
        m, n = X.shape

        # Initialize weights
        self.w = np.zeros(n)

        # Stochastic Gradient Descent
        for epoch in range(self.epochs):
            for i in range(m):
                if y[i] * np.dot(self.w, X[i]) < 1:
                    self.w = self.w + self.learning_rate * (self.C * y[i] * X[i] - self.w)
                else:
                    self.w = self.w - self.learning_rate * self.w

    def predict(self, X):
        """
        Make predictions using the trained model.

        Parameters:
        - X (numpy.ndarray): Test feature data.

        Returns:
        - numpy.ndarray: Predicted class labels.
        """
        # Add bias term to X
        X = np.column_stack((np.ones(len(X)), X))
        return np.sign(np.dot(X, self.w))


# Linear SVM Regressor from Scratch
class LinearSVMRegressor:
    def __init__(self, learning_rate=0.01, epochs=1000, C=1.0):
        """
        Initialize a Linear SVM Regressor.

        Parameters:
        - learning_rate (float): Learning rate for gradient descent.
        - epochs (int): Number of training epochs.
        - C (float): Regularization parameter.
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.C = C

    def fit(self, X, y):
        """
        Train the Linear SVM Regressor.

        Parameters:
        - X (numpy.ndarray): Training feature data.
        - y (numpy.ndarray): Training target data.
        """
        # Add bias term to X
        X = np.column_stack((np.ones(len(X)), X))
        m, n = X.shape

        # Initialize weights
        self.w = np.zeros(n)

        # Stochastic Gradient Descent
        for epoch in range(self.epochs):
            for i in range(m):
                error = y[i] - np.dot(self.w, X[i])
                if error != 0:
                    self.w = self.w + self.learning_rate * (self.C * error * X[i] - self.w)
                else:
                    self.w = self.w - self.learning_rate * self.w

    def predict(self, X):
        """
        Make predictions using the trained model.

        Parameters:
        - X (numpy.ndarray): Test feature data.

        Returns:
        - numpy.ndarray: Predicted target values.
        """
        # Add bias term to X
        X = np.column_stack((np.ones(len(X)), X))
        return np.dot(X, self.w)


# Instantiate the Linear SVM Classifier
svm_classifier = LinearSVMClassifier(learning_rate=0.01, epochs=1000, C=1.0)

# Train the model
svm_classifier.fit(X_train, np.sign(y_train))

# Make predictions on the test set
y_pred_clf = svm_classifier.predict(X_test)

# Visualize the results for classification
plt.scatter(X_test, y_test, color='black', label='True values')
plt.scatter(X_test, y_pred_clf, color='red', label='Predicted values')
plt.title('Linear SVM Classification from Scratch')
plt.xlabel('Median Income')
plt.ylabel('Above Median House Value (1) / Below Median (0)')
plt.legend()
plt.show()


# Instantiate the Linear SVM Regressor
svm_regressor = LinearSVMRegressor(learning_rate=0.01, epochs=1000, C=1.0)

# Train the model
svm_regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred_reg = svm_regressor.predict(X_test)

# Visualize the results for regression
plt.scatter(X_test, y_test, color='black', label='True values')
plt.scatter(X_test, y_pred_reg, color='red', label='Predicted values')
plt.title('Linear SVM Regression from Scratch')
plt.xlabel('Median Income')
plt.ylabel('Median House Value')
plt.legend()
plt.show()
