'''
Audio Signal Separation:

Dataset: Audio recordings with multiple sound sources.
Task: Separate different instruments or voices present in a mixed audio signal.

Medical Imaging:

Dataset: Functional Magnetic Resonance Imaging (fMRI) or Electroencephalography (EEG) data.
Task: Identify independent brain sources or components related to different cognitive processes.

Image Processing:

Dataset: Multispectral or hyperspectral images.
Task: Extract independent components representing distinct features or materials in the images.

Financial Time Series Analysis:

Dataset: Stock prices or financial time series data.
Task: Identify independent factors influencing the variations in financial data.

Communication Signal Processing:

Dataset: Mixed signals in communication channels.
Task: Separate independent sources in mixed signals, especially in scenarios like multiple microphones in speech processing.
Biological Data Analysis:

Dataset: Gene expression data.
Task: Identify independent gene expression patterns or regulatory pathways.

Environmental Monitoring:

Dataset: Sensor data from environmental monitoring stations.
Task: Extract independent components representing different environmental factors.

Sensory Data Processing:

Dataset: Multisensory data from sensors.
Task: Separate independent sources or features from combined sensory signals.

Network Traffic Analysis:

Dataset: Network traffic logs with multiple sources.
Task: Identify independent patterns or sources of network activity.

Social Sciences:

Dataset: Surveys or observational data with multiple influencing factors.
Task: Extract independent social or psychological factors affecting the observed outcomes.
'''



import numpy as np
import matplotlib.pyplot as plt

def center_data(X):
    """Center the data by subtracting the mean."""
    mean = np.mean(X, axis=1, keepdims=True)
    return X - mean

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def fastICA(X, max_iter=1000, tol=1e-4):
    """
    Perform FastICA to recover independent components from mixed signals.

    Parameters:
    - X: numpy array, shape (n_samples, n_features)
        Input data matrix containing mixed signals.
    - max_iter: int, optional (default=1000)
        Maximum number of iterations for the FastICA algorithm.
    - tol: float, optional (default=1e-4)
        Tolerance to declare convergence.

    Returns:
    - W: numpy array, shape (n_features, n_features)
        Demixing matrix.
    - S: numpy array, shape (n_samples, n_features)
        Estimated independent components.
    """

    # Center the data
    X = center_data(X)

    # Initialize random demixing matrix
    W = np.random.rand(X.shape[0], X.shape[0])

    for iteration in range(max_iter):
        # Compute the estimated sources
        S = np.dot(W, X)

        # Compute the contrast function and its gradient
        g = sigmoid(S)
        g_prime = 1 - 2 * g

        # Update the demixing matrix
        W_new = np.dot(g, S.T) / X.shape[1] - np.diag(np.mean(g_prime, axis=1)) @ W

        # Decorrelate the rows of the new demixing matrix
        U, _, Vt = np.linalg.svd(W_new)
        W_new = np.dot(U, Vt)

        # Check for convergence
        if np.max(np.abs(np.abs(np.diag(np.dot(W_new, W.T))) - 1)) < tol:
            break

        W = W_new

    S = np.dot(W, X)
    return W, S

# Generate synthetic data
np.random.seed(42)
t = np.linspace(0, 1, 1000)
s1 = np.sin(2 * np.pi * 5 * t)  # Signal 1
s2 = np.sign(np.sin(2 * np.pi * 10 * t))  # Signal 2
S_true = np.vstack((s1, s2))

# Mix the signals
A = np.array([[1, 1], [0.5, 2]])  # Mixing matrix
X = np.dot(A, S_true)

# Apply FastICA to recover independent components
W, S = fastICA(X)

# Plot original signals and recovered signals
plt.figure(figsize=(10, 4))

plt.subplot(2, 2, 1)
plt.title("Signal 1")
plt.plot(t, S_true[0, :])

plt.subplot(2, 2, 2)
plt.title("Signal 2")
plt.plot(t, S_true[1, :])

plt.subplot(2, 2, 3)
plt.title("Mixed Signal 1")
plt.plot(t, X[0, :])

plt.subplot(2, 2, 4)
plt.title("Mixed Signal 2")
plt.plot(t, X[1, :])

plt.tight_layout()
plt.show()


