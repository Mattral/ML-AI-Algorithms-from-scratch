# mlscratch Roadmap

Tracks the planned work by priority tier. Items move to `CHANGELOG.md` when shipped.

---

## P0 — Must-have (blocking next release)

- [ ] **Neural Networks module** (`mlscratch.neural`) — port all algorithms from
  `Neural Networks/` folder into the `src/` package with matching test files:
  - AutoEncoder
  - BoltzmannMachine
  - GenerativeAdversarialNetwork (GAN)
  - HopfieldNetwork
  - LSTM (LongShortTermMemory)
  - MLP Classification + Regression
  - RadialBasisFunctionNetworks
  - SelfAttentionMechanism
  - SimpleCNN
  - SimpleEncoderDecoder
  - SimpleRNN
  - SingleLayerPerceptron (Classification + Regression)
  - Transformer

- [ ] **Supervised module** (`mlscratch.supervised`) — port all from `Supervised/`:
  - DecisionTree
  - KNearestNeighbour
  - LassoRegression
  - LinearRegression
  - LogisticRegression
  - RandomForest
  - SVM (Support Vector Machine)
  - GradientBoosting
  - AdaBoost
  - Ridge / ElasticNet

- [ ] **Complete type annotations** across all 30+ source files

- [ ] **PyPI first release** — tag v0.1.0, trigger CI release pipeline

---

## P1 — Should-have (next minor version)

- [ ] **MkDocs documentation site** hosted on GitHub Pages
  - Quickstart page
  - API reference (auto-generated from docstrings via mkdocstrings)
  - Algorithm reference with equations
  - Benchmark results page

- [ ] **Colab quickstart notebooks** (`notebooks/`) — one per sub-package,
  runnable in < 2 minutes with no setup

- [ ] **Property-based tests** — add Hypothesis-based tests for numerical
  invariants (e.g. "for any valid input shape, predict() output shape matches")

- [ ] **Benchmark suite** (`benchmarks/`) — timing vs. scikit-learn equivalents
  on standard datasets (iris, boston, mnist subset)

- [ ] **Coverage gate raised to 80%** (currently 70%)

- [ ] **`conftest.py` fixtures used everywhere** — audit all test files and
  replace ad-hoc dataset construction with shared fixtures

- [ ] **Examples directory** — one runnable script per algorithm:
  ```
  examples/
    unsupervised_kmeans.py
    unsupervised_pca.py
    bayesian_gp.py
    reinforcement_dqn.py
    ...
  ```

---

## P2 — Nice-to-have (future)

- [ ] **`mlscratch.datasets`** — tiny built-in toy datasets (no sklearn needed):
  `make_blobs`, `make_moons`, `make_classification`, `load_iris`

- [ ] **`mlscratch.metrics`** — accuracy, precision, recall, F1, ROC-AUC,
  confusion matrix, MSE, MAE, R² — all pure NumPy

- [ ] **`mlscratch.preprocessing`** — StandardScaler, MinMaxScaler,
  LabelEncoder, OneHotEncoder, train_test_split

- [ ] **`mlscratch.pipeline`** — sklearn-compatible `Pipeline` and
  `cross_val_score`

- [ ] **GPU-optional back-end** — `import cupy as np` fallback when CUDA available,
  zero code changes in algorithm files

- [ ] **ONNX export** for trained model weights (neural networks)

- [ ] **Interactive visualisations** in notebooks — decision boundaries,
  training curves, attention maps

- [ ] **Citation file** (`CITATION.cff`) — make it easy to cite in academic work

- [ ] **Multilingual README** — Chinese, Arabic translations (community-driven)

---

## Completed

- [x] `mlscratch.unsupervised` — 9 algorithms, 100+ tests (v0.1.0)
- [x] `mlscratch.bayesian` — 7 algorithms, 150+ tests (v0.1.0)
- [x] `mlscratch.reinforcement` — 5 algorithms + utils, 120+ tests (v0.1.0)
- [x] Production `pyproject.toml` with full PyPI metadata
- [x] GitHub Actions CI (lint → test matrix → build → PyPI release)
- [x] `CONTRIBUTING.md`, `CHANGELOG.md`, `requirements.txt`
- [x] `conftest.py` shared test fixtures
- [x] PEP 561 typed package (`py.typed`)
- [x] CLI entry point (`python -m mlscratch`)
