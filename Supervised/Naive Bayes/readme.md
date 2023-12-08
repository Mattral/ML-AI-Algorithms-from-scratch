# Naive Bayes Classifier from Scratch

## Overview

The Naive Bayes Classifier is a probabilistic machine learning algorithm based on Bayes' theorem. It makes predictions by calculating the probability of each class given the input features and choosing the class with the highest probability. This README covers the implementation of a simple Naive Bayes Classifier from scratch using Python. It explains the workings of the algorithm, its pros and cons, and provides examples for better understanding.

## How Naive Bayes Classifier Works

### Mathematical Formulation

1. **Data Preprocessing:**
      - The algorithm assumes that features are conditionally independent given the class label. Preprocess the data by tokenizing and vectorizing the text features (e.g., using bag-of-words or TF-IDF).
   - Tokenize and vectorize text features into a bag-of-words representation. Given a document with features \(X_1, X_2, ..., X_n\), the probability of the document given the class \(P(X_1, X_2, ..., X_n | C)\) is calculated.

3. **Training:**
      - Estimate the probability of each class and the conditional probabilities of each feature given the class using the training data.
   - Calculate the prior probability of each class \(P(C)\) and the conditional probabilities of each feature given the class \(P(X_i | C)\) using the training data.
   - Use Bayes' theorem to calculate the posterior probability of each class \(P(C | X_1, X_2, ..., X_n)\).

   $$\[ P(C | X_1, X_2, ..., X_n) = \frac{P(C) \cdot P(X_1 | C) \cdot P(X_2 | C) \cdot ... \cdot P(X_n | C)}{P(X_1, X_2, ..., X_n)} \]$$

5. **Prediction:**
   - For a new data point, calculate the posterior probability for each class and choose the class with the highest probability as the predicted class.

   $$\[ \hat{y} = \arg\max_{c} P(C=c | X_1, X_2, ..., X_n) \]$$

## Pros and Cons

### Pros

- **Simple and Fast:** Naive Bayes is computationally efficient and easy to implement.
- **Works Well with Text Data:** Particularly effective for text classification tasks like spam filtering.
- **Handles Irrelevant Features:** It performs well even when irrelevant features are present.

### Cons

- **Assumption of Independence:** The "naive" assumption of feature independence may not hold in real-world scenarios.
- **Lack of Model Interpretability:** It doesn't provide insights into the relationships between features.
- **Sensitive to Outliers:** Outliers in the data can impact the results.

## Example Usage

```python
# Instantiate and train the Naive Bayes Classifier
nb_classifier = NaiveBayesClassifier()
nb_classifier.train(X_train, y_train)

# Make predictions on the test set
predictions = nb_classifier.predict(X_test)

# Evaluate the model
accuracy, class_report, confusion_mat = nb_classifier.evaluate(X_test, y_test)

# Display results
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:\n', class_report)
print('Confusion Matrix:\n', confusion_mat)
```
