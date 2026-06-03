import numpy as np

from mlscratch.supervised.logistic_regression import LogisticRegression


def test_logistic_regression_learns_binary_separation():
    """LogisticRegression must achieve ≥95% accuracy on clean binary data."""
    rng = np.random.default_rng(42)
    X = np.vstack([
        rng.normal(loc=-2.0, scale=0.5, size=(100, 2)),
        rng.normal(loc=2.0, scale=0.5, size=(100, 2)),
    ])
    y = np.concatenate([np.zeros(100), np.ones(100)])

    model = LogisticRegression(n_epochs=2000, learning_rate=0.1, batch_size=32, random_state=42)
    model.fit(X, y)
    accuracy = model.score(X, y)
    assert accuracy >= 0.95


def test_logistic_regression_predict_proba_shape():
    """predict_proba must return a probability for each sample."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 3))
    y = np.zeros(10)
    model = LogisticRegression(n_epochs=1)
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (10,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_logistic_regression_score_matches_accuracy():
    """score() must equal the fraction of correct binary predictions."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(float)
    model = LogisticRegression(n_epochs=1000, learning_rate=0.05, batch_size=10, random_state=1)
    model.fit(X, y)
    assert model.score(X, y) == np.mean(model.predict(X) == y)


def test_logistic_regression_invalid_labels_raise():
    """Non-binary labels should raise a ValueError."""
    X = np.ones((5, 2))
    y = np.array([0, 1, 2, 0, 1])
    model = LogisticRegression(n_epochs=1)
    try:
        model.fit(X, y)
    except ValueError as exc:
        assert "binary labels" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-binary labels")
