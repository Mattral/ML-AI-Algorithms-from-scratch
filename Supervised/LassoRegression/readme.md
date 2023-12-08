**Algorithm:**
Lasso Regression (Least Absolute Shrinkage and Selection Operator) is a linear regression technique with regularization. It adds a penalty term to the linear regression cost function to prevent overfitting and encourage feature selection.

The objective function for Lasso Regression is:

$$\[ J(\theta) = \text{MSE}(\theta) + \alpha \sum_{i=1}^{n} |\theta_i| \]$$

Here:
- \( J(\theta) \) is the cost function.
- \(\text{MSE}(\theta)\) is the Mean Squared Error, the traditional linear regression cost.
- \(\alpha\) is the regularization parameter controlling the strength of the regularization term.

The optimization involves updating the coefficients (\(\theta_i\)) using coordinate descent, where each coefficient is adjusted in turn while keeping others fixed.

**Uses:**
1. **Feature Selection:** Lasso Regression tends to shrink the coefficients of less important features to exactly zero, effectively performing feature selection.
2. **Regression with Regularization:** When dealing with high-dimensional data or multicollinearity, Lasso can be useful to prevent overfitting.

**Mathematical Expressions:**

*Objective Function:*
$$\[ J(\theta) = \text{MSE}(\theta) + \alpha \sum_{i=1}^{n} |\theta_i| \]$$

*Update Rule (Coordinate Descent):*
```math
\[ \theta_j^{new} = \text{soft\_threshold}(\rho_j, \alpha) \]$$
\[ \text{soft\_threshold}(x, \alpha) = \text{sign}(x) \cdot \max(|x| - \alpha, 0) \]
```

**Pros and Cons:**

**Pros:**
1. **Feature Selection:** Lasso can automatically select relevant features by driving some coefficients to zero.
2. **Simplicity:** Simple to implement and understand.
3. **Regularization:** Effective in preventing overfitting, especially in high-dimensional datasets.

**Cons:**
1. **Instability:** Lasso can be sensitive to multicollinearity.
2. **Non-Unique Solutions:** In cases of highly correlated features, Lasso might give non-unique solutions.
3. **Sensitive to Scaling:** Lasso's performance can be influenced by the scale of the features.
