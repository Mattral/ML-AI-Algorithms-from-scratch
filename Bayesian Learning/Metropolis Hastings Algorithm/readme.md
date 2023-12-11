# Metropolis-Hastings Algorithm

## Overview:
Metropolis-Hastings (MH) is a Markov Chain Monte Carlo (MCMC) algorithm used for sampling from a probability distribution. The algorithm allows generating samples from a target distribution, even if the distribution is complex or not directly samplable.

## Algorithm Steps:

1. **Initialization:**
   - Start with an initial state $\( x_0 \)$.

2. **Proposal:**
   - Propose a new state $\( x' \)$ from a proposal distribution $\( q(x' | x_t) \)$, which defines the probability of moving from state $\( x_t \) to \( x' \)$.

3. **Acceptance Probability:**
   - Calculate the acceptance probability $\( \alpha = \min\left(1, \frac{p(x')}{p(x_t)} \cdot \frac{q(x_t | x')}{q(x' | x_t)}\right) \)$, where $\( p(x) \)$ is the target distribution.

4. **Accept/Reject:**
   - Accept the new state $\( x' \)$ with probability $\( \alpha \)$, else stay at the current state $\( x_t \)$.

5. **Repeat:**
   - Repeat steps 2-4 for a predefined number of iterations.

## Mathematical Explanation:

- **Acceptance Ratio (Transition Probability):**
  $$\[ \alpha = \min\left(1, \frac{p(x')}{p(x_t)} \cdot \frac{q(x_t | x')}{q(x' | x_t)}\right) \]$$
  The acceptance ratio ensures that the Markov chain asymptotically samples from the target distribution.

## Pros and Cons:

**Pros:**
- Flexibility: Can be used for a wide range of distributions, even when direct sampling is challenging.
- Convergence: Asymptotically converges to the target distribution.

**Cons:**
- Autocorrelation: Generated samples may exhibit autocorrelation, requiring careful analysis.
- Tuning: Selection of the proposal distribution can impact performance, requiring tuning.

## Real-World Uses:

- **Bayesian Inference:** Estimate posterior distributions in Bayesian statistics.
- **Machine Learning:** Sampling from complex posterior distributions in probabilistic machine learning models.

## Note:
Metropolis-Hastings is foundational to many MCMC algorithms, providing a basis for more advanced techniques like Gibbs Sampling and Hamiltonian Monte Carlo.

## References:
- [Metropolis-Hastings Algorithm](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm)

