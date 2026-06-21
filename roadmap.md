# mlscratch Roadmap

Tracks the planned work by priority tier. Items move to `CHANGELOG.md` when shipped.

---

## P0 — Must-have (blocking next release)

- [ ] **Complete type annotations** across all 30+ source files

- [ ] **PyPI first release** — published as `scratchkit` (the `mlscratch` name was already taken on PyPI by an unrelated project); tag a release, trigger the CI release pipeline

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

---

## P2 — Nice-to-have (future)

- [ ] **`mlscratch.datasets`** — tiny built-in toy datasets (no sklearn needed):
  `make_blobs`, `make_moons`, `make_classification`, `load_iris`

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
- [x] `mlscratch.neural` — 13 architectures (perceptrons, autoencoders,
  RNN/LSTM, CNN, attention/Transformer, GAN, Hopfield, RBM, RBF network,
  complex-valued NN), 372 tests — implemented prior to v0.2.0 but missing
  from this roadmap/the changelog until now
- [x] Production `pyproject.toml` with full PyPI metadata
- [x] GitHub Actions CI (lint → test matrix → build → PyPI release)
- [x] `CONTRIBUTING.md`, `CHANGELOG.md`, `requirements.txt`
- [x] `conftest.py` shared test fixtures
- [x] PEP 561 typed package (`py.typed`)
- [x] CLI entry point (`python -m mlscratch`)
- [x] `mlscratch.supervised` — Linear/Ridge/Lasso/ElasticNet/Logistic, KNN,
  DecisionTree, RandomForest, kernel `SVC` (SMO), GradientBoosting, AdaBoost;
  236+ tests (v0.2.0)
- [x] `mlscratch.metrics` — classification + regression metrics, verified
  against scikit-learn (v0.2.0)
- [x] `mlscratch.preprocessing` — scalers, encoders, polynomial features,
  `train_test_split` (v0.2.0)
- [x] `examples/` directory — runnable end-to-end scripts (v0.2.0)
