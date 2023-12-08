'''
Data Loading and Preprocessing:

1. The program loads the Titanic dataset, which contains information about passengers, including whether they survived or not.
Selected features ('Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare') and the target variable ('Survived') are extracted.
The 'Sex' column, which contains categorical values ('female' and 'male'), is mapped to numerical values (0 and 1).

2. Data Splitting:
The data is split into training and testing sets using the train_test_split function from scikit-learn.

3. Feature Scaling:
Standardization is applied to the features using StandardScaler from scikit-learn.

4. Logistic Regression Model Training:
A logistic regression model is instantiated and trained on the standardized training data.
The fit method uses gradient descent to optimize weights and bias for the logistic regression model.
Loss is printed every 100 iterations to monitor the training progress.

5.Model Evaluation:
The trained model is used to make predictions on the standardized test set.
Confusion matrix and classification report are printed to evaluate model performance.

6. Accuracy Calculation:
The accuracy of the model on the test set is calculated and printed.

7.Confusion Matrix Plotting:
The confusion matrix is visualized using a heatmap.
'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.datasets import fetch_openml

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for iteration in range(self.num_iterations):
            z = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(z)
            errors = predictions - y

            # Update weights and bias using gradient descent
            self.weights -= self.learning_rate * (1 / X.shape[0]) * np.dot(X.T, errors)
            self.bias -= self.learning_rate * (1 / X.shape[0]) * np.sum(errors)

            # Print loss every 100 iterations
            if iteration % 100 == 0:
                loss = self.calculate_loss(predictions, y)
                print(f"Iteration {iteration}, Loss: {loss}")

    def calculate_loss(self, predictions, y):
        return -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))

    def predict(self, X):
        X = np.array(X)
        z = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(z)
        return (predictions >= 0.5).astype(int)


# Load Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Select relevant features and target
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = df['Survived']

# Convert 'Sex' to numerical values
X['Sex'] = X['Sex'].map({'female': 0, 'male': 1})

# Handle missing values
X['Age'].fillna(X['Age'].mean(), inplace=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Instantiate and train the logistic regression model
model = LogisticRegression(learning_rate=0.01, num_iterations=1000)
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
predictions_test = model.predict(X_test_scaled)

# Print confusion matrix and classification report
conf_matrix = confusion_matrix(y_test, predictions_test)
class_report = classification_report(y_test, predictions_test)
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# Calculate accuracy
accuracy = np.mean(predictions_test == y_test)
print(f"\nAccuracy on the test set: {accuracy}")

# Plot confusion matrix
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
