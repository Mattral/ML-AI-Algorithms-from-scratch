import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Define the true parameters of the Gaussian distribution
true_mean = 5.0
true_std = 1.0

# Generate synthetic data
np.random.seed(42)
data = np.random.normal(loc=true_mean, scale=true_std, size=100)

# Variational Inference implementation
def variational_inference(data, num_iterations=1000, learning_rate=0.01):
    # Initialize variational parameters
    mean_param = np.random.randn()
    std_param = np.random.randn()

    for iteration in range(num_iterations):
        # Update mean parameter
        mean_param += learning_rate * np.mean(data - mean_param)

        # Update standard deviation parameter
        std_param += learning_rate * (np.mean((data - mean_param)**2) - 1)

    return mean_param, np.exp(std_param)

# Perform variational inference
estimated_mean, estimated_std = variational_inference(data)

# Plot the results
x = np.linspace(3, 7, 100)
true_distribution = norm.pdf(x, loc=true_mean, scale=true_std)
estimated_distribution = norm.pdf(x, loc=estimated_mean, scale=estimated_std)

plt.plot(x, true_distribution, label='True Distribution', color='blue')
plt.plot(x, estimated_distribution, label='Estimated Distribution', linestyle='--', color='red')
plt.title('Variational Inference for Gaussian Distribution')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.show()

# Display the results
print(f'True Mean: {true_mean}, True Standard Deviation: {true_std}')
print(f'Estimated Mean: {estimated_mean}, Estimated Standard Deviation: {estimated_std}')
