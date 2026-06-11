# Contributing to mlscratch

Thank you for your interest in contributing! This guide explains everything you
need to go from zero to a merged pull request.

---

## Table of contents

1. [Philosophy](#philosophy)
2. [Branch & PR strategy](#branch--pr-strategy)
3. [Development setup](#development-setup)
4. [Coding standards](#coding-standards)
5. [Writing tests](#writing-tests)
6. [Adding a new algorithm](#adding-a-new-algorithm)
7. [PR checklist](#pr-checklist)
8. [Commit message format](#commit-message-format)
9. [Release process](#release-process)

---

## Philosophy

mlscratch has one rule: **pure NumPy only.**

Every algorithm must be understandable by reading the source file top-to-bottom.
If the reader needs to visit a second file to understand a concept, that is a
documentation problem, not a dependency problem. Heavy frameworks (PyTorch,
TensorFlow, scikit-learn, scipy) are never imported in `src/`.

---

## Branch & PR strategy

```
main          ← stable, always installable, tagged for PyPI releases
dev           ← integration branch, all PRs target here
feature/<name>← individual feature branches, branch off dev
fix/<name>    ← bug-fix branches
```

**Never push directly to `main`.** All changes go through a PR into `dev`.
`dev` is merged to `main` only at release time.

---

## Development setup

```bash
# 1. Fork and clone
git clone https://github.com/<your-fork>/ML-AI-Algorithms-from-scratch.git
cd ML-AI-Algorithms-from-scratch

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install in editable mode with all dev tools
pip install -e ".[dev]"

# 4. Verify the install
python -m mlscratch info

# 5. Run the full test suite
pytest

# 6. Run only fast tests (skip @pytest.mark.slow)
pytest -m "not slow"

# 7. Run tests for one sub-package
pytest tests/unsupervised/

# 8. Check coverage
pytest --cov=src/mlscratch --cov-report=term-missing

# 9. Lint
ruff check src/ tests/
black --check src/ tests/
```

---

## Coding standards

### Style
- Line length: **100 characters** (enforced by `ruff` + `black`).
- Formatter: `black` — run `black src/ tests/` before committing.
- Linter: `ruff` — zero tolerance on E/F/W/I/UP/B categories.

### Type annotations
All public functions and methods must have return-type annotations.
Parameter types are strongly encouraged. Use `from __future__ import annotations`
at the top of every source file.

```python
# Good
def fit(self, X: np.ndarray, y: np.ndarray) -> "MyAlgorithm":
    ...

# Bad — missing annotations
def fit(self, X, y):
    ...
```

### Docstrings
Module-level, class-level, and public-method docstrings are required.
Use the NumPy docstring style:

```python
def predict(self, X: np.ndarray) -> np.ndarray:
    """
    Predict class labels for samples in X.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)

    Returns
    -------
    labels : ndarray of shape (n_samples,)
    """
```

### No external ML imports in `src/`
```python
# Allowed
import numpy as np
from collections import deque
from itertools import combinations

# NEVER in src/mlscratch/
import sklearn
import torch
import tensorflow
import scipy
```

### Reproducibility
Every class that uses randomness must accept a `random_state: int | None = None`
parameter and use `np.random.default_rng(random_state)` internally.

---

## Writing tests

Tests live in `tests/<subpackage>/test_<algorithm>.py`.

### Three-tier structure (required for every algorithm)

```python
class TestAlgorithmBasic:
    """Shape contracts, return-type contracts, attribute existence."""

class TestAlgorithmCorrectness:
    """Analytically verifiable results — known ground truths."""

class TestAlgorithmEdgeCases:
    """Degenerate inputs: single sample, extreme parameters, etc."""
```

### Fixtures
Use the shared fixtures from `conftest.py` (`rng`, `small_X_y`, `blobs_X_y`,
etc.) instead of recreating datasets inside each test file.

### Assertions
Prefer `np.testing.assert_allclose` over `==` for floats.
Use `pytest.approx` for scalar comparisons.

### Coverage
New code must not drop overall coverage below **70%**.
Aim for ≥80% on new files. Check with:
```bash
pytest --cov=src/mlscratch --cov-report=term-missing
```

---

## Adding a new algorithm

1. **Create the source file** `src/mlscratch/<subpackage>/<algorithm>.py`
   - Start with a module docstring explaining the algorithm, equations, and references
   - Implement a class following the existing `fit / predict / fit_predict` API
   - Accept `random_state=None`
   - Pure NumPy only

2. **Export from `__init__.py`**
   Add the class to `src/mlscratch/<subpackage>/__init__.py`:
   ```python
   from .<algorithm> import MyAlgorithm  # noqa: F401
   # and add to __all__
   ```

3. **Write tests** `tests/<subpackage>/test_<algorithm>.py`
   Three classes, minimum 15 tests total.

4. **Add to CHANGELOG.md** under `[Unreleased] → Added`

5. **Add a usage example** `examples/<subpackage>_<algorithm>.py`

6. **Open a PR** into `dev` with the filled-out checklist below.

---

## PR checklist

Before requesting a review, confirm every item:

- [ ] `ruff check src/ tests/` passes with no errors
- [ ] `black --check src/ tests/` passes with no errors
- [ ] `pytest` passes with zero failures
- [ ] Coverage has not dropped (run `pytest --cov=src/mlscratch`)
- [ ] New public classes/functions have NumPy-style docstrings
- [ ] New public classes/functions have type annotations
- [ ] Algorithm source file has a module-level docstring with algorithm description and references
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] PR description explains *what* changed and *why*
- [ ] No external ML framework imports in `src/`
- [ ] `random_state` parameter added to all stochastic classes

---

## Commit message format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short description>

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `ci`, `perf`

Examples:
```
feat(unsupervised): add IsolationForest implementation
fix(bayesian): correct KL divergence sign in BNN
test(reinforcement): add TD3 delayed policy update assertion
docs(contributing): clarify no-external-imports rule
chore(ci): upgrade codecov action to v4
```

---

## Release process

Only maintainers release. The process is:

1. Merge `dev` → `main` via PR
2. Bump version in `pyproject.toml` (follow SemVer)
3. Update `CHANGELOG.md` — move `[Unreleased]` to `[X.Y.Z] — YYYY-MM-DD`
4. Commit: `chore(release): bump version to X.Y.Z`
5. Tag: `git tag vX.Y.Z && git push origin vX.Y.Z`
6. GitHub Actions `release` job automatically publishes to PyPI via Trusted Publishing

---

## Questions?

Open a [Discussion](https://github.com/Mattral/ML-AI-Algorithms-from-scratch/discussions)
or ping in an [Issue](https://github.com/Mattral/ML-AI-Algorithms-from-scratch/issues).
