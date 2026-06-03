# ML-AI-Algorithms-from-scratch

A structured, educational repository of from-scratch ML/AI/RL/Bayesian algorithms.

This project is evolving from a collection of standalone scripts into a clean, `pip`-installable Python package under `src/mlscratch/`.

---

## Current Status

- Standardized package layout under `src/mlscratch/`
- Verified supervised algorithms with `pytest`
- `README.md` updated to reflect current package state
- Added package-level implementations for:
  - `LinearRegression`
  - `LogisticRegression`
  - `LassoRegression`
  - `RidgeRegression`
  - `KNeighborsClassifier`
  - `DecisionTreeClassifier`
  - `RandomForestClassifier`
  - `GaussianNB`
  - `LinearSVMClassifier`
- Next implementation focus: unsupervised algorithms, beginning with `KMeans`

---

## Project Structure

```
ML-AI-Algorithms-from-scratch/
├── LICENSE
├── README.md
├── pyproject.toml
├── src/
│   └── mlscratch/
│       ├── __init__.py
│       ├── supervised/
│       │   ├── __init__.py
│       │   ├── linear_regression.py
│       │   ├── logistic_regression.py
│       │   ├── lasso_regression.py
│       │   ├── ridge_regression.py
│       │   ├── knn.py
│       │   ├── decision_tree.py
│       │   ├── random_forest.py
│       │   ├── naive_bayes.py
│       │   └── svm.py
│       └── unsupervised/  <- in progress
├── tests/
│   ├── conftest.py
│   ├── supervised/
│   │   ├── test_linear_regression.py
│   │   ├── test_logistic_regression.py
│   │   ├── test_lasso_regression.py
│   │   ├── test_ridge_regression.py
│   │   ├── test_knn.py
│   │   ├── test_decision_tree.py
│   │   ├── test_random_forest.py
│   │   ├── test_naive_bayes.py
│   │   └── test_svm.py
│   └── unsupervised/    <- coming next
```

---

## What This Repository Is For

This repo is intended as an educational reference for learners who want to understand the internal mechanics of algorithms, not as a production-ready library.

It prioritizes:

- clarity over micro-optimization
- math-first explanations
- algorithmic correctness through tests
- reproducible minimal examples

---

## Installation

```bash
python -m pip install -e .
python -m pip install -e .[dev]
```

The repository is designed to work with Python 3.10+.

---

## Quick Start

```python
from mlscratch.supervised import (
    OrdinaryLeastSquares,
    LogisticRegression,
    LassoRegression,
    RidgeRegression,
    KNeighborsClassifier,
    DecisionTreeClassifier,
    RandomForestClassifier,
    GaussianNB,
    LinearSVMClassifier,
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

## Testing

Run the supervised test suite:

```bash
python -m pytest tests/supervised -q
```

The repository uses `pytest` and is configured with `pytest-cov` for coverage reporting.

---

## Package Goals

The long-term goal is to make this repository a best-in-class educational reference by:

- standardizing module structure
- enforcing tests for correctness against `scikit-learn` baselines
- adding benchmark-driven performance comparisons
- documenting math and algorithmic intuition consistently

---

## Next Work

The next active task is to migrate unsupervised algorithms into `src/mlscratch/unsupervised/`, starting with a clean `KMeans` implementation and its test coverage.

After that, work will continue through the remaining `feedback.md` roadmap:

- unsupervised algorithms (`KMeans`, `PCA`, `GMM`, `DBSCAN`, `SOM`, `tSNE`)
- neural network modules
- reinforcement algorithms
- Bayesian algorithms

---

## Notes for Contributors

If you want to help improve this repository, focus on:

- adding `src/mlscratch/` modules for remaining algorithms
- matching the package template used by existing supervised implementations
- writing tests that compare output to `scikit-learn` or other reliable baselines
- keeping documentation concise and mathematically rigorous



