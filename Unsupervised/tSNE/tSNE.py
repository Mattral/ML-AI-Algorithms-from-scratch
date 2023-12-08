import numpy as np
import matplotlib.pyplot as plt

def compute_pairwise_distances(X):
    """
    Compute pairwise Euclidean distances between data points.

    Parameters:
    - X: Input data matrix.

    Returns:
    - distances: Pairwise distances matrix.
    """
    N = X.shape[0]
    distances = np.zeros((N, N))
    
    for i in range(N):
        for j in range(i + 1, N):
            distances[i, j] = np.linalg.norm(X[i] - X[j])
            distances[j, i] = distances[i, j]
    
    return distances

def compute_conditional_probabilities(distances, perplexity, epsilon=1e-8):
    """
    Compute conditional probabilities for t-SNE.

    Parameters:
    - distances: Pairwise distances matrix.
    - perplexity: Perplexity parameter.
    - epsilon: Small value to avoid division by zero.

    Returns:
    - P: Conditional probabilities matrix.
    """
    N = distances.shape[0]
    P = np.zeros((N, N))

    for i in range(N):
        # Compute conditional probabilities using binary search for better numerical stability
        lower_bound = -np.inf
        upper_bound = np.inf
        beta = 1.0

        while True:
            log_sum_Pi = -np.inf
            for j in range(N):
                if i != j:
                    log_sum_Pi = np.logaddexp(log_sum_Pi, -beta * distances[i, j])

            sum_Pi = np.exp(log_sum_Pi)
            sum_Pi += epsilon  # Avoid division by zero

            entropy = 0.0

            for j in range(N):
                if i != j:
                    log_Pij = -beta * distances[i, j] - log_sum_Pi
                    P[i, j] = np.exp(log_Pij)
                    entropy += P[i, j] * (log_Pij - np.log(sum_Pi))

            entropy_diff = np.log2(perplexity) - entropy

            if np.abs(entropy_diff) < 1e-5:
                break

            if entropy_diff > 0:
                lower_bound = beta
                if upper_bound == np.inf or upper_bound == -np.inf:
                    beta *= 2.0
                else:
                    beta = (beta + upper_bound) / 2.0
            else:
                upper_bound = beta
                if lower_bound == np.inf or lower_bound == -np.inf:
                    beta /= 2.0
                else:
                    beta = (beta + lower_bound) / 2.0

    return P




def compute_perplexity(P):
    """
    Compute perplexity of the distribution.

    Parameters:
    - P: Conditional probabilities matrix.

    Returns:
    - perplexity: Perplexity of the distribution.
    """
    entropy = -np.sum(P * np.log2(P + 1e-12))
    perplexity = 2 ** entropy
    return perplexity

def compute_grad(Y, P, Q):
    """
    Compute gradient of t-SNE.

    Parameters:
    - Y: Low-dimensional representation.
    - P: Conditional probabilities matrix.
    - Q: Low-dimensional similarities matrix.

    Returns:
    - grad: Gradient of t-SNE.
    """
    N, _ = Y.shape
    grad = np.zeros_like(Y)

    for i in range(N):
        diff = (P[i, :] - Q[i, :]) * (Y[i, :] - Y)
        grad[i, :] = np.sum(diff, axis=0)

    return grad

def tsne(X, num_dimensions=2, perplexity=30.0, num_iterations=1000, learning_rate=200.0):
    """
    t-SNE algorithm for dimensionality reduction.

    Parameters:
    - X: Input data matrix.
    - num_dimensions: Number of dimensions in the low-dimensional representation.
    - perplexity: Perplexity parameter for conditional probabilities.
    - num_iterations: Number of iterations.
    - learning_rate: Learning rate for gradient descent.

    Returns:
    - Y: Low-dimensional representation.
    """
    N, D = X.shape

    # Initialize low-dimensional representation randomly
    Y = np.random.randn(N, num_dimensions)

    # Compute pairwise distances
    distances = compute_pairwise_distances(X)

    for iteration in range(num_iterations):
        # Compute conditional probabilities
        P = compute_conditional_probabilities(distances, perplexity)

        # Compute low-dimensional similarities
        Q = 1.0 / (1.0 + compute_pairwise_distances(Y))

        # Compute gradient
        grad = compute_grad(Y, P, Q)

        # Update low-dimensional representation
        Y -= learning_rate * grad

        # Print progress
        if iteration % 100 == 0:
            cost = np.sum(P * np.log(P / Q))
            print(f"Iteration {iteration}/{num_iterations}, Cost: {cost}")

    return Y

# Generate synthetic data
np.random.seed(42)
X = np.random.randn(300, 50)

# Apply t-SNE
low_dimensional_representation = tsne(X, num_dimensions=2)

# Plot the low-dimensional representation
plt.scatter(low_dimensional_representation[:, 0], low_dimensional_representation[:, 1])
plt.title('t-SNE Visualization')
plt.show()
