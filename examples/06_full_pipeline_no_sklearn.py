"""
06_full_pipeline_no_sklearn.py
=================================
A complete, self-contained ML workflow built entirely from
mlscratch — preprocessing, model selection, an ensemble estimator,
and evaluation metrics — with zero scikit-learn dependency anywhere
in the pipeline. This is the shape of a typical real project using
this library end to end.

Part 1: classification
  raw data -> train/test split -> StandardScaler -> RandomForestClassifier
  -> classification_report

Part 2: regression
  raw data -> train/test split -> PolynomialFeatures -> LinearRegression
  -> RMSE / R^2

Run:
    python examples/06_full_pipeline_no_sklearn.py
"""

from _datasets import make_blobs, make_regression_line

from mlscratch.metrics import (
    accuracy_score,
    classification_report,
    r2_score,
    root_mean_squared_error,
)
from mlscratch.preprocessing import PolynomialFeatures, StandardScaler, train_test_split
from mlscratch.supervised import LinearRegression, RandomForestClassifier


def classification_pipeline() -> None:
    print("=" * 60)
    print("PART 1 — Classification pipeline")
    print("=" * 60)

    X, y = make_blobs(
        n_samples=400, centers=((-3, -3), (3, 3), (-3, 3), (3, -3)), cluster_std=1.8, random_state=0
    )

    # 1. Split BEFORE fitting any preprocessor, to avoid leaking test-set
    #    statistics into the scaler.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=0
    )

    # 2. Fit the scaler on train only, then apply it to both splits.
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Fit the model.
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
    model.fit(X_train_scaled, y_train)

    # 4. Evaluate with mlscratch.metrics.
    preds = model.predict(X_test_scaled)
    print(f"\nTest accuracy: {accuracy_score(y_test, preds):.3f}\n")
    print(classification_report(y_test, preds, target_names=["A", "B", "C", "D"]))


def regression_pipeline() -> None:
    print("\n" + "=" * 60)
    print("PART 2 — Regression pipeline (with polynomial features)")
    print("=" * 60)

    X, y = make_regression_line(n_samples=200, n_features=1, noise=3.0, random_state=1)
    # Bend the otherwise-linear target into a mild curve so that
    # polynomial expansion has something real to capture.
    y = y + 0.6 * X.ravel() ** 2

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    poly = PolynomialFeatures(degree=2, include_bias=False).fit(X_train)
    X_train_poly = poly.transform(X_train)
    X_test_poly = poly.transform(X_test)

    linear_model = LinearRegression().fit(X_train, y_train)
    poly_model = LinearRegression().fit(X_train_poly, y_train)

    print(f"\nFeature names used by the polynomial model: {poly.get_feature_names(['x'])}")
    print(f"\n{'model':<20}{'test RMSE':>12}{'test R^2':>12}")
    for name, preds in (
        ("plain linear", linear_model.predict(X_test)),
        ("degree-2 polynomial", poly_model.predict(X_test_poly)),
    ):
        rmse = root_mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        print(f"{name:<20}{rmse:>12.3f}{r2:>12.3f}")


if __name__ == "__main__":
    classification_pipeline()
    regression_pipeline()
