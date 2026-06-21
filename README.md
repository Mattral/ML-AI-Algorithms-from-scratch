# ML-AI-Algorithms-from-scratch

**60+ ML/AI/DL/RL/Bayesian algorithms implemented from scratch in NumPy — plus `mlscratch`, a pip-installable package with a consistent, scikit-learn-style API and 1,100+ tests.**

[![CI](https://github.com/Mattral/ML-AI-Algorithms-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/Mattral/ML-AI-Algorithms-from-scratch/actions)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Stars](https://img.shields.io/github/stars/Mattral/ML-AI-Algorithms-from-scratch?style=social)](https://github.com/Mattral/ML-AI-Algorithms-from-scratch/stargazers)

> **What's here:** readable, standalone implementations of algorithms you already know by name, written to show the math in code, not to be fast.
>
> **What's new:** `src/mlscratch/` — a pip-installable package with `fit()`/`predict()`/`transform()` APIs, full type hints, and a test suite that cross-checks correctness against scikit-learn wherever a reference implementation exists.

---

## What makes this different from the dozens of similar repos

There are many "ML from scratch" repos on GitHub. The honest differentiators here:

- **Bayesian methods are first-class.** Most from-scratch repos stop at supervised learning + neural nets. This one includes Bayesian Neural Networks, Gaussian Processes, Hidden Markov Models, Bayesian Networks, and Kalman Filters — algorithms most tutorials skip because they're harder to implement correctly.
- **RL goes beyond DQN.** DDPG, TD3, SAC, and PPO are included alongside tabular Q-Learning and DQN — non-trivial to implement correctly from scratch, and rare to see done well in a single repo.
- **The `src/mlscratch` package is real, not a wrapper.** Every estimator is implemented in pure NumPy — no calling out to scikit-learn at runtime. scikit-learn only appears in the *test suite*, as a correctness oracle, never as a dependency of the library itself.
- **Kernel SVM via real SMO, gradient boosting with proper Newton-step leaves, multiclass-native AdaBoost (SAMME.R)** — the ensemble/kernel methods aren't toy simplifications; several are verified to match scikit-learn's output to floating-point tolerance on real benchmarks.

---

## Quick start

### Browse the standalone scripts (no install needed)

```bash
git clone https://github.com/Mattral/ML-AI-Algorithms-from-scratch
cd ML-AI-Algorithms-from-scratch

pip install numpy matplotlib scikit-learn   # only deps, for the standalone scripts

python "Supervised/LinearRegression/linear_regression.py"
python "Neural Networks/Transformer/transformer.py"
python "Reinforcement/PPO/ppo.py"
```

### Use the package

```bash
pip install -e .                  # installs src/mlscratch in editable mode
# pip install -e ".[dev]"         # + pytest, ruff, black, mypy, for development

pytest tests/ -v                  # run the test suite
python -m mlscratch info          # package + sub-package summary
python -m mlscratch list supervised
```

```python
from mlscratch.supervised import RandomForestClassifier
from mlscratch.preprocessing import StandardScaler, train_test_split
from mlscratch.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)

scaler = StandardScaler().fit(X_train)
model = RandomForestClassifier(n_estimators=200, max_depth=6, oob_score=True)
model.fit(scaler.transform(X_train), y_train)

print(f"OOB score: {model.oob_score_:.3f}")
print(classification_report(y_test, model.predict(scaler.transform(X_test))))
```

See [`examples/`](examples/) for six runnable end-to-end scripts covering decision trees, random forests, kernel SVMs, gradient boosting, AdaBoost, and a full no-sklearn classification + regression pipeline.

---

## What's implemented

### `mlscratch` package (`src/mlscratch/`)

| Sub-package | Contents | Tests |
|---|---|---|
| `mlscratch.supervised` | Linear/Ridge/Lasso/ElasticNet/Logistic regression, KNN, **DecisionTree** (classifier + regressor), **RandomForest** (bagging + OOB scoring), kernel **SVC** (SMO; linear/poly/rbf/sigmoid, one-vs-rest multiclass), **GradientBoosting** (classifier + regressor, squared/absolute-error loss), **AdaBoost** (SAMME / SAMME.R, multiclass-native) | 162 |
| `mlscratch.unsupervised` | K-Means++, K-Medoids, DBSCAN, Agglomerative Clustering, PCA, t-SNE, FastICA, Gaussian Mixture Model (EM), Apriori | 120 |
| `mlscratch.bayesian` | Naive Bayes (Gaussian/Multinomial/Bernoulli), Bayesian Linear Regression, Bayesian Network, Bayesian Neural Network (mean-field VI), Gaussian Process Regression, Hidden Markov Model, Kalman Filter | 171 |
| `mlscratch.reinforcement` | Q-Learning, Double Q-Learning, DQN (Double + Dueling + PER), DDPG, TD3, PPO (GAE-λ), SAC, plus shared `GridWorld`/`ReplayBuffer`/`PrioritizedReplayBuffer` utilities | 218 |
| `mlscratch.neural` | Single/Multi-Layer Perceptron, Autoencoder (vanilla/denoising/variational), RNN/LSTM/Encoder-Decoder, a small CNN (Conv2D/Pool/BatchNorm), Attention + Transformer encoder, GAN, Hopfield Network, Restricted Boltzmann Machine, RBF Network, Complex-Valued NN | 372 |
| `mlscratch.metrics` | accuracy/precision/recall/F1, confusion matrix, `classification_report`, ROC/AUC, log loss, MSE/RMSE/MAE/MAPE, R², explained variance — every metric checked against scikit-learn | 48 |
| `mlscratch.preprocessing` | StandardScaler, MinMaxScaler, RobustScaler, Normalizer, LabelEncoder, OneHotEncoder, PolynomialFeatures, `train_test_split` (with stratification) | 62 |

**1,153 tests total.** A handful (~18) fail under the newest NumPy/SciPy releases in this environment due to upstream API drift in unrelated modules (Bayesian networks, reinforcement learning buffers, ICA) — tracked as known issues, not part of this release's scope.

### Standalone scripts (original, by category)

These are the original from-scratch scripts the package above was distilled from — browse them like a reference, run them directly, no install required.

- **`Supervised/`** — Linear/Ridge/Lasso Regression, Logistic Regression, k-NN, Decision Trees, Random Forest, Naive Bayes, SVM
- **`Unsupervised/`** — K-Means++, K-Medoids, DBSCAN, Hierarchical Clustering, PCA, t-SNE, ICA, Gaussian Mixture Model, EM, Self-Organising Map, Apriori
- **`Neural Networks/`** — Single/Multi-Layer Perceptron, Simple RNN, LSTM, Simple CNN, Encoder-Decoder, Self-Attention, Transformer, Autoencoder, GAN, Boltzmann Machine, Hopfield Network, RBF Networks
- **`Reinforcement/`** — Q-Learning, DQN, DDPG, PPO, SAC
- **`Bayesian Learning/`** — Bayesian Inference, Bayesian Linear Regression, Bayesian Network, Bayesian Neural Networks, Gibbs Sampling, Metropolis-Hastings, Variational Inference

---

## Design philosophy

Every implementation applies the same principles:

- Explicit loops over vectorised one-liners when clarity improves
- Model logic, loss computation, and parameter updates in separate functions
- The package layer (`src/mlscratch`) calls **only** NumPy at runtime — scikit-learn appears solely in the test suite, as a correctness oracle
- Short files: most standalone scripts are 100–300 lines; package modules favor one well-documented class per concern

**This trades raw performance for readability and correctness-by-inspection. That's intentional.**

If you're looking for production-speed implementations, use scikit-learn, PyTorch, or JAX. If you want to read the math in code form — or verify it against a reference implementation in the test suite — this is the repo.

---

## Recommended learning path

If you're working through this systematically:

1. Start with `Supervised/LinearRegression` (or `mlscratch.supervised.LinearRegression`) — the simplest possible end-to-end example
2. Move to `LogisticRegression` — same structure, adds sigmoid + cross-entropy
3. Then `DecisionTreeClassifier` → `RandomForestClassifier` → `GradientBoostingClassifier`/`AdaBoostClassifier` — the tree-ensemble family, building on a shared CART implementation
4. Then `Neural Networks/SingleLayerPerceptron` → `MultiLayerPerceptron` — backprop from first principles
5. Then any of: Unsupervised (PCA → GMM → t-SNE), Reinforcement (Q-Learning → DQN → PPO/SAC), or Bayesian (Naive Bayes → Bayesian Linear Regression → Variational Inference)

Each folder/module is reasonably self-contained — jump to any algorithm without reading the others first.

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
├── src/mlscratch/           Pip-installable package
│   ├── supervised/          Linear models, KNN, trees, ensembles, kernel SVM
│   ├── unsupervised/        Clustering, dimensionality reduction, association rules
│   ├── bayesian/            Naive Bayes, BLR, BNN, GP, HMM, Bayesian Networks, Kalman
│   ├── reinforcement/       Q-Learning, DQN, DDPG, TD3, PPO, SAC
│   ├── neural/              Perceptrons, autoencoders, RNN/CNN, attention, GAN, ...
│   ├── metrics/             Classification & regression evaluation metrics
│   └── preprocessing/       Scalers, encoders, polynomial features, train_test_split
│
├── examples/                Runnable end-to-end scripts (no sklearn at runtime)
├── tests/                   1,153 tests, mirroring the src/mlscratch layout
├── docs/                    Roadmap (MkDocs site planned, see roadmap.md)
├── pyproject.toml           Package metadata + deps
├── CHANGELOG.md             Keep-a-Changelog formatted release history
├── roadmap.md               P0 / P1 / P2 backlog
├── .github/workflows/       CI: lint → test matrix → build → PyPI release
└── README.md
```

---

## Contributing

The most useful contributions right now:

- **Add a standalone script** for an algorithm not yet covered (check the folder first)
- **Port a standalone script** into `src/mlscratch` with a matching test file in `tests/`
- **Fix a numerical issue** — some implementations have known edge cases under newer NumPy/SciPy releases (see the known-issues note above; open an issue or PR)

Standard flow: fork → branch → PR. CI runs `ruff`, `black --check`, and the full `pytest` suite on every PR. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full guide, and [`roadmap.md`](roadmap.md) for what's planned next.

---

## Honest scope

The standalone scripts under `Supervised/`, `Neural Networks/`, etc. are a **learning reference**, not a performance library: some use toy datasets, a few have hardcoded hyperparameters to keep the code short, and none are tuned for speed at scale.

The `src/mlscratch` package is more rigorous (typed, tested, cross-checked against scikit-learn) but is still pure-Python/NumPy — it will not outrun scikit-learn or XGBoost on large datasets, and that was never the goal. The public API is stabilising but may still change between minor versions before a 1.0 release; pin a version if you're building on top of it.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
