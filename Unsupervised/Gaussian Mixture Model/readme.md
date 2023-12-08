## Gaussian Mixture Model (GMM)

### Overview:

The Gaussian Mixture Model is a probabilistic model that represents a mixture of multiple Gaussian distributions. It is widely used for clustering and density estimation tasks. The model assumes that the data is generated from a mixture of several Gaussian distributions, each with its own set of parameters.

### Mathematical Expressions:

1. **Probability Density Function (PDF) for a single Gaussian distribution:**

   ![pdf](https://latex.codecogs.com/svg.latex?p(x)%20=%20\frac{1}{(2\pi)^{D/2}|\Sigma|^{1/2}}%20\exp\left(-\frac{1}{2}(x%20-%20\mu)^T\Sigma^{-1}(x%20-%20\mu)\right))

   Where:
   - \(x\) is the data point.
   - \(\mu\) is the mean vector.
   - \(\Sigma\) is the covariance matrix.
   - \(D\) is the dimensionality of the data.

2. **Mixture Model Probability Density Function:**

   ![mixture_model](https://latex.codecogs.com/svg.latex?p(x)%20=%20\sum_{i=1}^{K}%20\pi_i%20\cdot%20p_i(x))

   Where:
   - \(K\) is the number of components in the mixture.
   - \(\pi_i\) is the weight of the \(i\)-th component.
   - \(p_i(x)\) is the PDF of the \(i\)-th Gaussian component.

### Algorithm Steps:

1. **Initialization:**
   - Initialize parameters (means, covariances, and weights) either randomly or using a different method.

2. **Expectation Step:**
   - Compute the probability (responsibility) of each data point belonging to each Gaussian component using the current parameters.

   ![expectation_step](https://latex.codecogs.com/svg.latex?r_{i}(n)%20=%20\frac{\pi_i%20\cdot%20\mathcal{N}(x_n|\mu_i,%20\Sigma_i)}{\sum_{j=1}^{K}%20\pi_j%20\cdot%20\mathcal{N}(x_n|\mu_j,%20\Sigma_j)})

3. **Maximization Step:**
   - Update the parameters (means, covariances, and weights) based on the computed responsibilities.

   ![maximization_step_mu](https://latex.codecogs.com/svg.latex?\mu_i%20=%20\frac{\sum_{n=1}^{N}%20r_i(n)%20\cdot%20x_n}{\sum_{n=1}^{N}%20r_i(n)})

   ![maximization_step_sigma](https://latex.codecogs.com/svg.latex?\Sigma_i%20=%20\frac{\sum_{n=1}^{N}%20r_i(n)%20\cdot%20(x_n%20-%20\mu_i)(x_n%20-%20\mu_i)^T}{\sum_{n=1}^{N}%20r_i(n)})

   ![maximization_step_pi](https://latex.codecogs.com/svg.latex?\pi_i%20=%20\frac{\sum_{n=1}^{N}%20r_i(n)}{N})

4. **Convergence:**
   - Repeat the Expectation-Maximization steps until convergence (parameters stop changing significantly).

### Uses:

- **Clustering:** GMM can be used for soft clustering, where each data point is assigned a probability of belonging to each cluster.

- **Density Estimation:** GMM provides a generative model for the underlying distribution of the data.

- **Anomaly Detection:** GMM can be used to model the normal distribution of data and identify outliers.

### Pros and Cons:

**Pros:**
- **Flexibility:** Can model complex distributions.
- **Soft Clustering:** Provides probabilities for data points belonging to each cluster.
- **Density Estimation:** Useful for estimating the underlying distribution of data.

**Cons:**
- **Sensitivity to Initialization:** Performance may depend on the initial parameter values.
- **Computational Complexity:** Training can be computationally expensive, especially for high-dimensional data.
- **Assumes Gaussian Distribution:** May not perform well if the data doesn't follow a Gaussian distribution.

