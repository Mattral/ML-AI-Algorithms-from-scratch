import unittest
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from DecisionTreeRegression import DecisionTreeRegressor

class TestDecisionTreeRegressor(unittest.TestCase):
    def setUp(self):
        # Load the California housing dataset
        california_housing = fetch_california_housing()
        X = california_housing.data
        y = california_housing.target
        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Create and train the decision tree regressor
        self.dt_regressor = DecisionTreeRegressor(max_depth=3)
        self.dt_regressor.fit(self.X_train, self.y_train)

    def test_predict(self):
        # Ensure that predict returns a list of predicted values
        y_pred = self.dt_regressor.predict(self.X_test)
        self.assertIsInstance(y_pred, list)

    def test_mean_squared_error(self):
        # Ensure that mean_squared_error is a non-negative value
        mse = self.dt_regressor.mean_squared_error(self.y_test, self.dt_regressor.predict(self.X_test))
        self.assertGreaterEqual(mse, 0)

if __name__ == '__main__':
    unittest.main()
