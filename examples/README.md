# mlscratch examples

Runnable, self-contained scripts demonstrating the supervised estimators,
`mlscratch.metrics`, and `mlscratch.preprocessing` working together.

Every script uses only `numpy` and `mlscratch` at runtime — none of them
require scikit-learn, matplotlib, or any network access. Synthetic
datasets are generated locally by [`_datasets.py`](_datasets.py).

## Running

From the repository root, with `mlscratch` installed (`pip install -e .`):

```bash
python examples/01_decision_tree_classification.py
python examples/02_random_forest_vs_single_tree.py
python examples/03_svm_kernels_comparison.py
python examples/04_gradient_boosting_regression.py
python examples/05_adaboost_multiclass.py
python examples/06_full_pipeline_no_sklearn.py
```

Each script prints its results straight to the console — no plotting
dependency required.

## What each script covers

| Script | Algorithm(s) | Highlights |
|---|---|---|
| `01_decision_tree_classification.py` | `DecisionTreeClassifier` | gini vs entropy, `max_depth` overfitting curve, `classification_report` |
| `02_random_forest_vs_single_tree.py` | `DecisionTreeClassifier`, `RandomForestClassifier` | bagging variance reduction, out-of-bag scoring, `feature_importances_` |
| `03_svm_kernels_comparison.py` | `SVC` | linear vs poly vs rbf kernels, effect of `C`, one-vs-rest multiclass |
| `04_gradient_boosting_regression.py` | `GradientBoostingRegressor` | `staged_predict` convergence, `squared_error` vs `absolute_error` robustness |
| `05_adaboost_multiclass.py` | `AdaBoostClassifier` | `SAMME` vs `SAMME.R`, round-by-round accuracy, weak-learner depth |
| `06_full_pipeline_no_sklearn.py` | `RandomForestClassifier`, `LinearRegression` | the full shape of a real project: split → preprocess → fit → evaluate |

## Design notes

- **No data leakage**: every script splits the data *before* fitting any
  preprocessor (scaler, encoder, polynomial expander), then fits the
  preprocessor on the training split only.
- **`_datasets.py`** is a tiny, dependency-free synthetic-data module
  (blobs, two-moons, noisy lines, noisy sine waves) shared by every
  example, so the scripts stay focused on the algorithm being
  demonstrated rather than on data wrangling.
- Each script is independent — read the ones relevant to you, in any
  order.
