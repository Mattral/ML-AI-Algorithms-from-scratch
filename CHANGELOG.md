# Changelog

All notable changes to **mlscratch** are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Version numbers follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- Neural Networks module (`mlscratch.neural`):
  AutoEncoder, BoltzmannMachine, GAN, HopfieldNetwork, LSTM, MLP (classification
  & regression), RBF Networks, Self-Attention, CNN, Encoder-Decoder, RNN,
  Single-Layer Perceptron, Transformer
- Type stubs for all public APIs
- Property-based tests with Hypothesis
- MkDocs documentation site
- Colab quickstart notebooks

---

## [0.1.0] — 2026-06-10

### Added

#### Package infrastructure
- `pyproject.toml` — PEP 621-compliant, full classifiers, optional-dep groups
  (`dev`, `docs`, `notebooks`, `all`), PyPI Trusted Publishing ready
- `src/mlscratch/__init__.py` — top-level package, `__version__` via
  `importlib.metadata`
- `src/mlscratch/py.typed` — PEP 561 typed-package marker
- `src/mlscratch/__main__.py` — CLI: `python -m mlscratch info | list | version`
- `.github/workflows/ci.yml` — lint → test matrix (3.10/3.11/3.12) → build →
  PyPI release on tag push
- `conftest.py` — shared pytest fixtures: `rng`, `small_X_y`, `blobs_X_y`,
  `regression_X_y`, `tiny_grid`, `disc_env`, `cont_env`
- `requirements.txt` — pinned dev dependencies
- `CONTRIBUTING.md` — branch strategy, PR checklist, coding standards
- `roadmap.md` — P0 / P1 / P2 backlog

#### `mlscratch.unsupervised`
- `KMeans` — Lloyd's algorithm with k-means++ initialisation
- `DBSCAN` — density-based spatial clustering, core/border/noise labelling
- `PCA` — eigen-decomposition, `fit_transform`, `inverse_transform`,
  explained variance ratio
- `GaussianMixtureModel` — EM with log-sum-exp stability, convergence detection
- `AgglomerativeClustering` — single / complete / average / Ward linkages
- `KMedoids` — PAM algorithm, actual-datapoint medoids
- `Apriori` — association rule mining, support / confidence / lift
- `FastICA` — logcosh and exp contrast functions, whitening
- `TSNE` — perplexity binary search, early exaggeration, momentum GD

#### `mlscratch.bayesian`
- `GaussianNB`, `MultinomialNB`, `BernoulliNB` — log-space, Laplace smoothing
- `BayesianLinearRegression` — conjugate Gaussian posterior, evidence approx.
- `GaussianProcessRegressor` + kernels: `RBFKernel`, `Matern52Kernel`,
  `LinearKernel`, `PeriodicKernel` — Cholesky solve, posterior sampling
- `HiddenMarkovModel` — scaled forward-backward, Viterbi, Baum-Welch EM
- `BayesianNeuralNetwork` — mean-field VI, local reparameterisation, KL penalty
- `BayesianNetwork` — DAG with CPTs, variable elimination, ancestral sampling
- `KalmanFilter` — predict/update, log-likelihood, RTS smoother

#### `mlscratch.reinforcement`
- Shared utilities: `GridWorld`, `ContinuousEnv`, `DiscreteEnv`,
  `ReplayBuffer`, `PrioritizedReplayBuffer` (sum-tree), `MLP` (Adam backprop),
  `OrnsteinUhlenbeckNoise`, `GaussianNoise`
- `QLearning`, `DoubleQLearning`, `LinearQLearning` — tabular TD control
- `DQN` — Double DQN + Dueling + Prioritised Replay, soft/hard target sync
- `DDPG` — actor-critic, OU/Gaussian noise, soft target update
- `TD3` — twin critics, delayed policy update, target policy smoothing
- `PPO` — GAE-λ, clip + KL variants, discrete and continuous action spaces
- `SAC` — squashed Gaussian, twin soft Q-critics, automatic entropy tuning

### Tests
- 370+ tests across `tests/unsupervised/`, `tests/bayesian/`,
  `tests/reinforcement/`
- Three-tier structure per algorithm: Basic API · Correctness · Edge Cases
- Analytically verifiable assertions (Bellman update, d-separation, Kalman
  smoother monotonicity, GPR interpolation, etc.)

---

[Unreleased]: https://github.com/Mattral/ML-AI-Algorithms-from-scratch/compare/v0.1.0...HEAD
[0.1.0]:      https://github.com/Mattral/ML-AI-Algorithms-from-scratch/releases/tag/v0.1.0
