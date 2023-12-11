# Gibbs Sampling

## Overview

Gibbs Sampling is a Markov Chain Monte Carlo (MCMC) algorithm used for generating samples from a joint distribution. It is particularly useful when the joint distribution is complex and sampling directly from it is challenging. Gibbs Sampling iteratively samples from the conditional distributions of each variable given the current values of the other variables.

## Algorithm Description

1. **Initialization:** Start with initial values for each variable.

2. **Iterations:**
   - For each variable, update its value based on the current values of the other variables using their conditional distributions.
   - Repeat this process for a predefined number of iterations.

3. **Output:** The generated samples represent an approximation of the joint distribution.

## Mathematical Explanation

Let \( (x, y) \) be the variables of interest. The algorithm updates each variable based on the conditional distribution:
```math
\[ x \sim \mathcal{N}(y, 1.0) \]
\[ y \sim \mathcal{N}(x, 1.0) \]
```
Here, $\( \mathcal{N}(\mu, \sigma) \)$ denotes the normal distribution with mean $\( \mu \)$ and standard deviation $\( \sigma \)$.

## Pros and Cons

### Pros:
- Gibbs Sampling is easy to implement and often more straightforward than other MCMC methods.
- It is effective in sampling from high-dimensional and complex joint distributions.

### Cons:
- Convergence can be slow, especially when variables are highly correlated.
- The algorithm may not perform well if variables have non-linear dependencies.

## Use Cases

- **Bayesian Inference:** Gibbs Sampling is commonly used in Bayesian statistics for estimating posterior distributions.
- **Image Processing:** It has applications in image analysis, where joint distributions can be complex.
- **Machine Learning:** Gibbs Sampling is employed in various machine learning models, especially those involving latent variables.

## Example

Consider the following Python code snippet for a basic Gibbs Sampling implementation:

```python
import numpy as np

def target_distribution(x, y):
    return np.exp(-(x**2 + y**2))

def gibbs_sampling(iterations):
    x = 0.0
    y = 0.0
    samples = []

    for _ in range(iterations):
        x = np.random.normal(loc=y, scale=1.0)
        y = np.random.normal(loc=x, scale=1.0)
        samples.append((x, y))

    return np.array(samples)

# Number of iterations
iterations = 1000

# Run Gibbs Sampling
samples = gibbs_sampling(iterations)

# Display results
print("Gibbs Sampling Results:")
print(f"Mean of x: {np.mean(samples[:, 0]):.4f}")
print(f"Mean of y: {np.mean(samples[:, 1]):.4f}")
```
