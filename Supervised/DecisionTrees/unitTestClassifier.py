import unittest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from DecisionTreeClassification import DecisionTreeClassifier

class TestDecisionTreeClassifier(unittest.TestCase):
    def setUp(self):
        # Load the Iris dataset
        iris = load_iris()
        X = iris.data
        y = iris.target
        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Create and train the decision tree classifier
        self.dt_classifier = DecisionTreeClassifier()
        self.dt_classifier.fit(self.X_train, self.y_train)

    def test_predict(self):
        # Ensure that predict returns a list of predicted labels
        y_pred = self.dt_classifier.predict(self.X_test)
        self.assertIsInstance(y_pred, list)

    def test_accuracy(self):
        # Ensure that accuracy is between 0 and 1
        correct_predictions = sum(1 for pred, true in zip(self.dt_classifier.predict(self.X_test), self.y_test) if pred == true)
        accuracy = correct_predictions / len(self.y_test)
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)

if __name__ == '__main__':
    unittest.main()
