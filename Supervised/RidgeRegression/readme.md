**Algorithm:**
Ridge Regression is a linear regression technique with regularization. It adds a penalty term to the linear regression cost function to prevent overfitting and handle multicollinearity.

The objective function for Ridge Regression is:

$$\[ J(\theta) = \text{MSE}(\theta) + \alpha \sum_{i=1}^{n} \theta_i^2 \]$$

Here:
- \( J(\theta) \) is the cost function.
- \(\text{MSE}(\theta)\) is the Mean Squared Error, the traditional linear regression cost.
- \(\alpha\) is the regularization parameter controlling the strength of the regularization term.

The optimization involves updating the coefficients (\(\theta_i\)) using the closed-form solution or gradient descent.

**Uses:**
1. **Regularization:** Ridge Regression is effective in preventing overfitting, especially in cases of high-dimensional data or multicollinearity.
2. **Regression with Regularization:** It is useful when traditional linear regression might fail due to the presence of correlated features.

**Mathematical Expressions:**

*Objective Function:*
$$\[ J(\theta) = \text{MSE}(\theta) + \alpha \sum_{i=1}^{n} \theta_i^2 \]$$

*Closed-Form Solution:*
$$\[ \theta = (X^T X + \alpha I)^{-1} X^T y \]$$

**Pros and Cons:**

**Pros:**
1. **Regularization:** Ridge Regression effectively handles multicollinearity and prevents overfitting.
2. **Stability:** It provides a stable solution even when features are highly correlated.
3. **Closed-Form Solution:** Ridge Regression has a closed-form solution, which can be computationally efficient.

**Cons:**
1. **Feature Shrinkage, not Selection:** Ridge Regression tends to shrink the coefficients towards zero but rarely sets them exactly to zero, making it less effective for feature selection.
2. **Sensitivity to Scale:** Ridge Regression is sensitive to the scale of features, so feature scaling is important.
