import unittest
import numpy as np
from kmeanAlgo import kmeans

class TestKMeans(unittest.TestCase):

    def test_kmeans(self):
        # Generate a simple dataset for testing
        data = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])

        # Test kmeans with k=2
        k = 2
        centroids, labels = kmeans(data, k)
        
        # Ensure the number of centroids and labels match the expected values
        self.assertEqual(len(centroids), k)
        self.assertEqual(len(labels), len(data))

        # Ensure that the labels are integers and within the expected range
        print("Label Types:", [type(label) for label in labels])  # Add this line
        invalid_labels = [label for label in labels if not isinstance(label, np.int64)]  # Adjust the condition
        self.assertTrue(all(isinstance(label, np.int64) for label in labels), f"Invalid labels: {invalid_labels}")
        self.assertTrue(all(label >= 0 and label < k for label in labels))

    def test_kmeans_convergence(self):
        # Generate a dataset that should converge quickly
        data = np.concatenate([np.random.randn(100, 2), np.random.randn(100, 2) + [5, 5]])

        # Test kmeans with a low max_iters and high tolerance
        k = 2
        max_iters = 10
        tol = 1e-1
        centroids, labels = kmeans(data, k, max_iters=max_iters, tol=tol)
        
        # Ensure that the algorithm converges within the specified iterations
        self.assertTrue(len(set(labels)) == k)

if __name__ == '__main__':
    unittest.main()
