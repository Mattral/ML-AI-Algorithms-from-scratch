import numpy as np
import matplotlib.pyplot as plt

class SOM:
    def __init__(self, input_size, output_size, learning_rate=0.1, sigma=1.0):
        """
        Initialize the Self-Organizing Map (SOM).

        Parameters:
        - input_size: Number of features in the input data.
        - output_size: Tuple representing the grid size of the SOM (e.g., (5, 5) for a 5x5 grid).
        - learning_rate: Learning rate for updating the weights.
        - sigma: Standard deviation of the neighborhood function.
        """
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.sigma = sigma

        # Initialize weights randomly
        self.weights = np.random.rand(output_size[0], output_size[1], input_size)

    def find_best_match(self, input_vector):
        """
        Find the Best Matching Unit (BMU) in the SOM for a given input vector.

        Parameters:
        - input_vector: Input vector to find the BMU for.

        Returns:
        - bmu_indices: Indices of the BMU in the SOM grid.
        """
        # Calculate distances between input vector and SOM neurons' weights
        distances = np.linalg.norm(self.weights - input_vector, axis=2)
        # Find indices of the neuron with the closest weight
        bmu_indices = np.unravel_index(np.argmin(distances, axis=None), distances.shape)
        return bmu_indices

    def update_weights(self, input_vector, bmu_indices, iteration):
        """
        Update the weights of the SOM neurons based on the input vector and learning rate.

        Parameters:
        - input_vector: Input vector used for updating weights.
        - bmu_indices: Indices of the Best Matching Unit.
        - iteration: Current iteration/epoch of training.
        """
        # Calculate the influence of each neuron in the neighborhood
        influence = np.exp(-((np.arange(self.output_size[0]) - bmu_indices[0])**2 +
                             (np.arange(self.output_size[1]) - bmu_indices[1])**2) / (2 * (self.sigma**2)))
        # Update the weights of neighboring neurons
        learning_rate = self.learning_rate * np.exp(-iteration / 1000)
        delta_weights = learning_rate * influence[:, np.newaxis, np.newaxis] * (input_vector - self.weights)
        self.weights += delta_weights

    def train(self, data, epochs):
        """
        Train the SOM on the input data.

        Parameters:
        - data: Input data for training.
        - epochs: Number of training epochs.
        """
        for epoch in range(epochs):
            for i in range(data.shape[0]):
                input_vector = data[i, :]
                bmu_indices = self.find_best_match(input_vector)
                self.update_weights(input_vector, bmu_indices, epoch)

    def visualize(self, data):
        """
        Visualize the trained SOM and input data.

        Parameters:
        - data: Input data for visualization.
        """
        # Visualize the SOM and data
        plt.plot(data[:, 0], data[:, 1], 'bo', label='Data Points')
        plt.scatter(self.weights[:, :, 0], self.weights[:, :, 1], marker='x', s=100, c='r', label='SOM Neurons')
        plt.legend()
        plt.title('2D Self-Organizing Map')
        plt.show()

# Example Usage:
# Generate random 2D data for demonstration
data_points = np.random.rand(100, 2)

# Create and train a 2D SOM
input_size = data_points.shape[1]
output_size = (5, 5)  # 5x5 grid of neurons
som = SOM(input_size, output_size)
som.train(data_points, epochs=1000)

# Visualize the SOM and data
som.visualize(data_points)
