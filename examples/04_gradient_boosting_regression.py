"""
04_gradient_boosting_regression.py
====================================
GradientBoostingRegressor on a noisy sine wave: shows the training
loss decreasing stage-by-stage (via staged_predict), the effect of
the learning rate / n_estimators trade-off, and how the
'absolute_error' loss resists gross outliers far better than
'squared_error'.

Run:
    python examples/04_gradient_boosting_regression.py
"""

import numpy as np
from _datasets import make_sine_regression

from mlscratch.metrics import mean_absolute_error, r2_score
from mlscratch.preprocessing import train_test_split
from mlscratch.supervised import GradientBoostingRegressor


def main() -> None:
    X, y = make_sine_regression(n_samples=300, noise=0.4, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    print("=== Effect of n_estimators (learning_rate=0.1, max_depth=3) ===")
    for n_estimators in (1, 10, 50, 200):
        model = GradientBoostingRegressor(
            n_estimators=n_estimators, learning_rate=0.1, max_depth=3, random_state=0
        )
        model.fit(X_train, y_train)
        r2 = r2_score(y_test, model.predict(X_test))
        print(f"  n_estimators={n_estimators:<4} test R^2: {r2:.3f}")

    print("\n=== Training loss decreases monotonically, stage by stage ===")
    model = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0
    )
    model.fit(X_train, y_train)
    checkpoints = [0, 9, 24, 49, 99]
    for i in checkpoints:
        print(f"  stage {i + 1:<4} train MSE: {model.train_score_[i]:.4f}")

    print("\n=== Robustness to outliers: squared_error vs absolute_error ===")
    y_train_outliers = y_train.copy()
    rng = np.random.default_rng(0)
    outlier_idx = rng.choice(len(y_train_outliers), 15, replace=False)
    y_train_outliers[outlier_idx] += (
        rng.choice([-1, 1], 15) * 40.0
    )  # gross outliers, ~5% of training rows

    for loss in ("squared_error", "absolute_error"):
        model = GradientBoostingRegressor(n_estimators=100, max_depth=3, loss=loss, random_state=0)
        model.fit(X_train, y_train_outliers)
        mae = mean_absolute_error(y_test, model.predict(X_test))
        print(f"  loss={loss:<14} test MAE (on clean test data): {mae:.3f}")
    print("  -> squared_error chases the outliers and distorts the whole fit;")
    print("     absolute_error's median-leaf updates barely notice them.")


if __name__ == "__main__":
    main()
