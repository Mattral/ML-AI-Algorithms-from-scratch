import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class NaiveBayesClassifier:
    def __init__(self):
        self.model = None
        self.vectorizer = None

    def train(self, X_train, y_train):
        """
        Train the Naive Bayes Classifier.

        Parameters:
        - X_train (pd.Series): Training features (e.g., email text).
        - y_train (pd.Series): Training labels (1 for spam, 0 for ham).
        """
        # Create a pipeline with CountVectorizer and Multinomial Naive Bayes
        self.vectorizer = CountVectorizer()
        self.model = MultinomialNB()

        # Fit the model
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_train_vectorized, y_train)

    def predict(self, X_test):
        """
        Make predictions using the trained model.

        Parameters:
        - X_test (pd.Series): Test features (e.g., email text).

        Returns:
        - predictions (pd.Series): Predicted labels.
        """
        X_test_vectorized = self.vectorizer.transform(X_test)
        predictions = self.model.predict(X_test_vectorized)
        return predictions

    def evaluate(self, X_test, y_test):
        """
        Evaluate the performance of the model on the test set.

        Parameters:
        - X_test (pd.Series): Test features (e.g., email text).
        - y_test (pd.Series): True labels for the test set.

        Returns:
        - accuracy (float): Accuracy of the model.
        - classification_report (str): Classification report.
        - confusion_matrix (pd.DataFrame): Confusion matrix.
        """
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        class_report = classification_report(y_test, predictions)
        confusion_mat = confusion_matrix(y_test, predictions)
        return accuracy, class_report, pd.DataFrame(confusion_mat, columns=['Predicted Ham', 'Predicted Spam'], index=['Actual Ham', 'Actual Spam'])

# Example usage
# Assuming you have a DataFrame 'df' with 'email_text' and 'label' columns
# Adjust the following code accordingly to fit your dataset

# Load the dataset
df = pd.read_csv('spam_ham_dataset.csv')

# Separate features (X) and labels (y)
X = df['text']  # Replace 'email_text' with the correct column name ('text' in this case)
y = df['label_num']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Instantiate and train the Naive Bayes Classifier
nb_classifier = NaiveBayesClassifier()
nb_classifier.train(X_train, y_train)

# Evaluate the model
accuracy, class_report, confusion_mat = nb_classifier.evaluate(X_test, y_test)

# Display results
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:\n', class_report)
print('Confusion Matrix:\n', confusion_mat)
