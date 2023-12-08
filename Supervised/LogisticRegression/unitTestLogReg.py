import unittest
import numpy as np
from LogisticRegressionAlgo import LogisticRegression

class TestLogisticRegression(unittest.TestCase):
    def setUp(self):
        # Generate synthetic data for binary classification
        np.random.seed(42)
        self.X_synthetic = 2 * np.random.rand(100, 2)
        self.y_synthetic = (self.X_synthetic[:, 0] + self.X_synthetic[:, 1] > 2).astype(int)

        # Create and train the logistic regression model
        self.model = LogisticRegression(learning_rate=0.01, num_iterations=1000)
        self.model.fit(self.X_synthetic, self.y_synthetic)

    def test_predict(self):
        # Test if predictions are of the correct shape
        predictions_synthetic = self.model.predict(self.X_synthetic)
        self.assertEqual(predictions_synthetic.shape[0], self.y_synthetic.shape[0])


    def test_loss_history(self):
        # Test if the loss history is non-empty and decreasing
        self.assertGreater(len(self.model.loss_history), 0)
        self.assertTrue(all(self.model.loss_history[i] >= self.model.loss_history[i+1] for i in range(len(self.model.loss_history)-1)))

if __name__ == '__main__':
    unittest.main()
