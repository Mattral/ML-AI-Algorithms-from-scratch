# mlscratch

**Pure-NumPy from-scratch implementations of ML / AI / RL / Bayesian algorithms.**

[![CI](https://github.com/Mattral/ML-AI-Algorithms-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/Mattral/ML-AI-Algorithms-from-scratch/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/mlscratch?color=blue)](https://pypi.org/project/mlscratch/)
[![Python](https://img.shields.io/pypi/pyversions/mlscratch)](https://pypi.org/project/mlscratch/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](https://github.com/Mattral/ML-AI-Algorithms-from-scratch/blob/main/LICENSE)
[![Coverage](https://codecov.io/gh/Mattral/ML-AI-Algorithms-from-scratch/branch/main/graph/badge.svg)](https://codecov.io/gh/Mattral/ML-AI-Algorithms-from-scratch)

---

## What is mlscratch?

mlscratch demystifies the algorithms that power modern AI by building each one
from mathematical first principles — **no PyTorch, no TensorFlow, no
scikit-learn**. Just `numpy` and the maths.

Every class follows the familiar `fit / predict / fit_transform` API so the
implementations are immediately comparable to their framework counterparts.

---

## Install

```bash
pip install mlscratch                  # core (numpy only)
pip install "mlscratch[dev]"           # + pytest, ruff, black, mypy
pip install "mlscratch[all]"           # + docs + notebooks
```

---

## 30-second quickstart

```python
from mlscratch.unsupervised import KMeans, PCA
from mlscratch.bayesian import GaussianNB
from mlscratch.reinforcement import QLearning, GridWorld
import numpy as np

# --- Clustering ---
rng = np.random.default_rng(0)
X = np.vstack([rng.normal(c, 0.4, (30, 2)) for c in [[0,0],[6,0],[3,5]]])
km = KMeans(n_clusters=3, random_state=0).fit(X)
print(km.labels_[:5])   # [0, 0, 0, 0, 0]

# --- Dimensionality reduction ---
pca = PCA(n_components=2).fit_transform(X)
print(pca.shape)         # (90, 2)

# --- Classification ---
y = np.repeat([0, 1, 2], 30)
gnb = GaussianNB().fit(X, y)
print(gnb.predict(X[:3]))  # [0, 0, 0]

# --- Reinforcement Learning ---
env = GridWorld(size=4)
agent = QLearning(n_states=16, n_actions=4, random_state=0)
agent.train(env, n_episodes=500)
print(f"Mean reward (last 100): {np.mean(agent.episode_rewards_[-100:]):.2f}")
```

---

## Modules

| Module | Algorithms |
|---|---|
| `mlscratch.unsupervised` | KMeans, DBSCAN, PCA, GMM, HAC, KMedoids, Apriori, FastICA, t-SNE |
| `mlscratch.bayesian` | GaussianNB, MultinomialNB, BernoulliNB, Bayesian Linear Regression, Gaussian Process, HMM, BNN, Bayesian Network, Kalman Filter |
| `mlscratch.reinforcement` | Q-Learning, Double Q-Learning, DQN (Double/Dueling/PER), DDPG, TD3, PPO, SAC |
| `mlscratch.supervised` | Linear/Logistic/Lasso/Ridge Regression, KNN, Decision Tree, Random Forest, SVM, AdaBoost, Gradient Boosting *(coming soon)* |
| `mlscratch.neural` | MLP, CNN, RNN, LSTM, Transformer, GAN, AutoEncoder, and more *(coming soon)* |

---

## CLI

```bash
python -m mlscratch info          # version + sub-package summary
python -m mlscratch list          # all available algorithm classes
python -m mlscratch list bayesian # algorithms in one sub-package
```

---

## Design principles

**Pure NumPy** — the only runtime dependency.
Every algorithm is readable top-to-bottom as a standalone file.

**Numerically careful** — log-space computations, Cholesky solves, scaled
forward-backward, variance smoothing; the same precautions used in production
libraries.

**sklearn-compatible API** — `fit`, `predict`, `fit_transform`,
`fit_predict`, `random_state`. Drop-in for exploration and education.

**Fully tested** — 370+ tests with three-tier structure: API contracts,
analytically verifiable correctness, and edge cases.
