# Changelog

All notable changes to **mlscratch** are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Version numbers follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- Type stubs for all public APIs
- Property-based tests with Hypothesis
- MkDocs documentation site
- Colab quickstart notebooks

---

## [0.2.0] — 2026-06-20

### Documentation
- **Corrected a stale planning record**: `mlscratch.neural` (13 architectures —
  perceptrons, autoencoders, RNN/LSTM, a small CNN, attention/Transformer,
  GAN, Hopfield network, RBM, RBF network, plus a bonus complex-valued NN —
  372 tests) was already fully implemented and tested in this codebase but
  had never been recorded in `CHANGELOG.md` or `roadmap.md`, both of which
  still listed it under "planned". Fixed here; see `mlscratch.neural`'s
  module docstring for the full architecture list.
- Rewrote `README.md`, which had drifted out of sync with the package
  (missing modules, stale "in progress" framing for code that already
  shipped, an inaccurate algorithm list)

### Changed
- `mlscratch.supervised` — extracted the input-validation logic that had
  been copy-pasted near-verbatim across `decision_tree.py`, `random_forest.py`,
  `svm.py`, `gradient_boosting.py`, and `adaboost.py` into one shared
  internal module, `mlscratch.supervised._validation`. Every estimator now
  raises identically-worded errors for the same class of mistake by
  construction rather than by convention; covered directly by
  `tests/supervised/test_validation.py` in addition to each estimator's
  existing edge-case tests. Pure refactor — no behavioural change.

### Added

#### `mlscratch.supervised` — tree-based & kernel models
- `DecisionTreeClassifier` / `DecisionTreeRegressor` — CART with gini/entropy/MSE
  criteria, a fully vectorised O(n log n)-per-feature split search, per-sample
  `sample_weight` support, `predict_proba`, `feature_importances_`, and `apply()`
  for leaf introspection (used internally by the new ensembles)
- `RandomForestClassifier` / `RandomForestRegressor` — bootstrap + random-subspace
  bagging, soft-voting `predict_proba`, out-of-bag scoring (`oob_score=True`)
- `SVC` — kernel Support Vector Classifier (linear / poly / rbf / sigmoid /
  custom callable) trained from scratch via Sequential Minimal Optimization
  (Platt's SMO); natively binary with transparent one-vs-rest multiclass.
  Replaces the previous linear-only `LinearSVMClassifier` placeholder.
- `GradientBoostingClassifier` (binary, binomial deviance) /
  `GradientBoostingRegressor` (`squared_error`, `absolute_error`) — Friedman's
  TreeBoost with proper Newton-step / median leaf refinement, `staged_predict`,
  `train_score_`, `feature_importances_`
- `AdaBoostClassifier` — `SAMME` and `SAMME.R`, natively multiclass
  (Zhu et al., 2009), `staged_predict`, `estimator_weights_`/`estimator_errors_`

#### `mlscratch.metrics` (new module)
- Classification: `accuracy_score`, `precision_score`, `recall_score`,
  `f1_score`, `precision_recall_fscore_support`, `confusion_matrix`,
  `classification_report`, `roc_curve`, `roc_auc_score`, `log_loss`
- Regression: `mean_squared_error`, `root_mean_squared_error`,
  `mean_absolute_error`, `mean_absolute_percentage_error`, `r2_score`,
  `explained_variance_score`
- Every metric verified against scikit-learn to floating-point tolerance

#### `mlscratch.preprocessing` (new module)
- Scalers: `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `Normalizer`
- Encoders: `LabelEncoder`, `OneHotEncoder`
- `PolynomialFeatures` — polynomial / interaction feature expansion
- `train_test_split` — with optional class-stratified splitting

#### `examples/`
- Runnable end-to-end scripts demonstrating the new estimators and the
  `metrics` / `preprocessing` modules together, including a from-scratch-only
  classification and regression pipeline with no scikit-learn dependency

### Fixed
- `mlscratch.supervised.knn` — `KNeighboursClassifier`/`Regressor` now validate
  input shapes in `fit`/`predict` (raising `ValueError`/`RuntimeError` instead
  of failing later with a confusing index error), gained a `score()` method,
  and are now also importable under the American spelling
  (`KNeighborsClassifier`, `KNeighborsRegressor`)
- `pyproject.toml` — `scikit-learn` was used as a correctness oracle by 20+
  test files but was never declared as a dependency anywhere, so a clean
  `pip install -e ".[dev]"` followed by `pytest` would fail with
  `ModuleNotFoundError`. Added to the `dev` extra (still never imported by
  `src/mlscratch` itself)
- `python -m mlscratch info` / `list` — the CLI's sub-package registry was
  hardcoded to 4 entries and silently omitted `mlscratch.neural` (already
  shipped) and the new `mlscratch.metrics` / `mlscratch.preprocessing`

### Tests
- 236+ new tests across `tests/supervised/`, `tests/metrics/`,
  `tests/preprocessing/`, continuing the Basic API · Correctness · Edge
  Cases structure, with correctness assertions cross-checked against
  scikit-learn wherever an equivalent reference implementation exists

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

[Unreleased]: https://github.com/Mattral/ML-AI-Algorithms-from-scratch/compare/v0.2.0...HEAD
[0.2.0]:      https://github.com/Mattral/ML-AI-Algorithms-from-scratch/compare/v0.1.0...v0.2.0
[0.1.0]:      https://github.com/Mattral/ML-AI-Algorithms-from-scratch/releases/tag/v0.1.0
