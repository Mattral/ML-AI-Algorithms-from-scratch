import unittest
import numpy as np
from LinearRegressionAlgo import LinearRegression  # Adjust the import statement based on your actual filename

class TestLinearRegression(unittest.TestCase):
    def setUp(self):
        # Set up synthetic data for testing
        np.random.seed(42)
        self.X_synthetic = 2 * np.random.rand(100, 1)
        self.y_synthetic = 4 + 3 * self.X_synthetic + np.random.randn(100, 1)

        # Set up the model for testing
        self.model = LinearRegression(learning_rate=0.01, num_iterations=1000)

    def test_fit_predict(self):
        # Test the fit and predict methods
        self.model.fit(self.X_synthetic, self.y_synthetic)
        predictions_synthetic = self.model.predict(self.X_synthetic)

        # Check if the model parameters are not None after fitting
        self.assertIsNotNone(self.model.weights)
        self.assertIsNotNone(self.model.bias)

        # Check if the predictions have the correct shape
        #self.assertEqual(predictions_synthetic.shape, self.y_synthetic.shape)
        self.assertEqual(predictions_synthetic.shape, (100, 1))


        # Additional assertions based on your expectations
        self.assertAlmostEqual(self.model.weights[0], 3.0, delta=1e-1)
        self.assertAlmostEqual(self.model.bias, 4.0, delta=1e-1)

    def test_calculate_loss(self):
        # Test the calculate_loss method
        
        self.model.fit(self.X_synthetic, self.y_synthetic.flatten())

        loss = self.model.calculate_loss(self.X_synthetic, self.y_synthetic)

        # Check if the loss is a non-negative number
        self.assertGreaterEqual(loss, 0)

        # Additional assertions based on your expectations
        self.assertAlmostEqual(loss, 0, delta=1e5)

if __name__ == '__main__':
    unittest.main()
    print("Test Passed")
