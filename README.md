# ML-AI-Algorithms-from-scratch

**50+ ML/AI/DL/RL algorithms implemented from scratch in NumPy — plus a growing `src/mlscratch` pip-installable package.**

[![CI](https://github.com/Mattral/ML-AI-Algorithms-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/Mattral/ML-AI-Algorithms-from-scratch/actions)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Stars](https://img.shields.io/github/stars/Mattral/ML-AI-Algorithms-from-scratch?style=social)](https://github.com/Mattral/ML-AI-Algorithms-from-scratch/stargazers)

> **What's here:** readable, standalone implementations of algorithms you already know by name, written to show the math in code, not to be fast.
>
> **What's new:** `src/mlscratch/` — a growing pip-installable package extracting the best implementations with unit tests, so you can `import mlscratch` and inspect or extend them programmatically.

---

## What makes this different from the dozens of similar repos

There are many "ML from scratch" repos on GitHub. The honest differentiators here:

- **Bayesian methods are first-class.** Most from-scratch repos stop at supervised + neural nets. This one includes Bayesian Neural Networks, Variational Inference, Gibbs Sampling, and Metropolis-Hastings — algorithms that most tutorials skip because they're harder to implement.
- **RL goes beyond DQN.** DDPG, SAC, and PPO are included alongside Q-Learning and DQN. These are non-trivial to implement correctly from scratch.
- **The `src/mlscratch` package evolution.** The standalone-script phase was the foundation. The current work refactors the best implementations into a proper Python package with tests — meaning you can now `pip install -e .` and run `import mlscratch.supervised.linear_regression` with a consistent API rather than hunting through folders.

---

## Quick start

### Browse the standalone scripts (no install needed)

```bash
git clone https://github.com/Mattral/ML-AI-Algorithms-from-scratch
cd ML-AI-Algorithms-from-scratch

pip install numpy matplotlib scikit-learn  # only deps

# Run any standalone script directly:
python "Supervised/LinearRegression/linear_regression.py"
python "Neural Networks/Transformer/transformer.py"
python "Reinforcement/PPO/ppo.py"
```

### Use the package (new)

```bash
pip install -e .   # installs src/mlscratch in editable mode

python -c "from mlscratch.supervised import LinearRegression; print('ok')"

# Run unit tests
pytest tests/ -v
```

---

## What's implemented

### Supervised Learning (`Supervised/`)

Linear Regression · Ridge Regression · Lasso Regression · Logistic Regression · k-Nearest Neighbours · Decision Trees · Random Forest · Naive Bayes · Support Vector Machines

### Unsupervised Learning (`Unsupervised/`)

K-Means++ · K-Medoids · DBSCAN · Hierarchical Clustering · PCA · t-SNE · ICA · Gaussian Mixture Model · Expectation-Maximization · Self-Organising Map · Apriori

### Neural Networks (`Neural Networks/`)

Single-Layer Perceptron · Multi-Layer Perceptron (Classification + Regression) · Simple RNN · LSTM · Simple CNN · Encoder-Decoder · Self-Attention · Transformer · Autoencoder · GAN · Boltzmann Machine · Hopfield Network · Radial Basis Function Networks

### Reinforcement Learning (`Reinforcement/`)

Q-Learning · Deep Q-Network (DQN) · Deep Deterministic Policy Gradients (DDPG) · Proximal Policy Optimisation (PPO) · Soft Actor-Critic (SAC)

### Bayesian Learning (`Bayesian Learning/`)

Bayesian Inference · Bayesian Linear Regression · Bayesian Network · Bayesian Neural Networks · Gibbs Sampling · Metropolis-Hastings · Variational Inference

---

## Design philosophy

Every implementation applies the same principles:

- Explicit loops over vectorised one-liners when clarity improves
- Model logic, loss computation, and parameter updates in separate functions
- No high-level ML libraries — only NumPy, basic Python, and matplotlib for plots
- Short files: most implementations are 100–300 lines

**This trades performance for readability. That's intentional.**

If you're looking for production-speed implementations, use scikit-learn, PyTorch, or JAX. If you want to read the math in code form, this is the repo.

---

## Recommended learning path

If you're working through this systematically:

1. Start with `Supervised/LinearRegression` — the simplest possible end-to-end example
2. Move to `Supervised/LogisticRegression` — same structure, adds sigmoid + cross-entropy
3. Then `Neural Networks/SingleLayerPerceptron` — backprop from first principles
4. Then `Neural Networks/MultiLayerPerceptron` — stack the layers
5. Then any of: Unsupervised (PCA → GMM → t-SNE), Reinforcement (Q-Learning → DQN → PPO), or Bayesian (BayesianInference → VariationalInference)

Each folder is self-contained. You can jump to any algorithm without reading the others first.

---

## `src/mlscratch` — the package layer (ongoing)

The top-level category folders (`Supervised/`, `Neural Networks/`, etc.) are the original standalone scripts — browse them like a reference, run them directly.

`src/mlscratch/` is an active refactor: taking the clearest implementations from those folders and packaging them with:

- Consistent sklearn-style API (`fit()`, `predict()`, `transform()`)
- Unit tests in `tests/`
- A `pyproject.toml` for `pip install -e .`

**Current status:** Supervised and Unsupervised algorithms are the most complete in the package. Neural nets, RL, and Bayesian methods are progressively being added. The standalone scripts remain the primary reference — the package layer is additive, not a replacement.

---

## Repository layout

```
ML-AI-Algorithms-from-scratch/
│
├── Supervised/              Standalone scripts: LinearRegression, SVM, etc.
├── Unsupervised/            Standalone scripts: KMeans++, DBSCAN, t-SNE, etc.
├── Neural Networks/         Standalone scripts: MLP, LSTM, Transformer, GAN, etc.
├── Reinforcement/           Standalone scripts: DQN, DDPG, PPO, SAC, etc.
├── Bayesian Learning/       Standalone scripts: BNN, VI, MCMC, etc.
│
├── src/mlscratch/           Pip-installable package (ongoing refactor)
│   ├── supervised/
│   ├── unsupervised/
│   ├── neural/
│   ├── reinforcement/
│   └── bayesian/
│
├── tests/                   Unit tests for src/mlscratch
├── pyproject.toml           Package metadata + deps
├── .github/workflows/       CI
└── README.md
```

---

## Contributing

The most useful contributions right now:

- **Add a standalone script** for an algorithm not yet covered (check the folder first)
- **Port a standalone script** into `src/mlscratch` with a matching test in `tests/`
- **Fix a numerical issue** — some older implementations have known edge cases (open an issue)

Standard flow: fork → branch → PR. CI runs `pytest tests/` on every PR.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) if it exists, otherwise the above is the full guide.

---

## Honest scope

This is a **learning reference**, not a performance library. The implementations prioritise code you can read over code that runs fast. Some scripts use toy datasets; a few have hardcoded hyperparameters to keep the code short. If you run them on real data at scale, they will be slow.

The `src/mlscratch` package is a work in progress. The API may change between commits. Pin a commit hash if you're building something on top of it.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
