import unittest
import numpy as np
from sklearn.preprocessing import StandardScaler
from LassoRegression import LassoRegression  
class TestLassoRegression(unittest.TestCase):
    def setUp(self):
        # Setting up a small dataset to use in tests
        self.X = np.array([[1, 2], [3, 4], [5, 6]])
        self.y = np.array([1, 3, 5])
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        # Create an instance of LassoRegression
        self.model = LassoRegression(alpha=0.1, max_iterations=1000, tol=1e-4)

    def test_initialization(self):
        """ Test the initialization of the LassoRegression model. """
        self.assertEqual(self.model.alpha, 0.1)
        self.assertEqual(self.model.max_iterations, 1000)
        self.assertEqual(self.model.tol, 1e-4)
        self.assertIsNone(self.model.theta)

    def test_fit(self):
        """ Test fitting the model correctly adjusts the theta attribute. """
        self.model.fit(self.X_scaled, self.y)
        self.assertIsNotNone(self.model.theta)  # Check that theta is updated
        self.assertEqual(len(self.model.theta), self.X_scaled.shape[1] + 1)  # Check theta length

    def test_predict(self):
        """ Test predictions are returned as expected. """
        self.model.fit(self.X_scaled, self.y)
        predictions = self.model.predict(self.X_scaled)
        self.assertEqual(predictions.shape[0], self.y.shape[0])  # Check prediction size

    def test_soft_threshold(self):
        """ Test the soft threshold function operates as expected. """
        result_positive = self.model.soft_threshold(0.5, 0.2)
        result_negative = self.model.soft_threshold(-0.5, 0.2)
        self.assertEqual(result_positive, 0.3)
        self.assertEqual(result_negative, -0.3)

if __name__ == '__main__':
    unittest.main()
